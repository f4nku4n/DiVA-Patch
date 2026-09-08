from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F

from attack_methods.WhiteBoxPatch import MaskedPGD, MaskedAutoPGD, LaVAN, LOAP


@dataclass
class GradientQuery:
    images: list
    true_label: int
    target_label: int
    targeted: bool
    objective: str = "ce"
    variable: object = None


@dataclass
class GradientResult:
    logits: torch.Tensor
    objectives: torch.Tensor
    gradient: object = None


class WhiteBoxGradientEvaluator:
    """Batch differentiable queries whose source images may have different sizes."""

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _objectives(logits, query):
        true = torch.full((len(logits),), query.true_label,
                          dtype=torch.long, device=logits.device)
        target = torch.full((len(logits),),
                            query.true_label if query.target_label is None else query.target_label,
                            dtype=torch.long, device=logits.device)
        if query.objective == "margin":
            rows = torch.arange(len(logits), device=logits.device)
            if query.targeted:
                return logits[rows, target] - logits[rows, true]
            other = logits.clone(); other[rows, true] = -torch.inf
            return other.max(1).values - logits[rows, true]
        if query.objective == "lavan":
            return (F.cross_entropy(logits, target, reduction="none")
                    - F.cross_entropy(logits, true, reduction="none"))
        labels = target if query.targeted else true
        ce = F.cross_entropy(logits, labels, reduction="none")
        return -ce if query.targeted else ce

    def evaluate(self, queries):
        flat_images = [image for query in queries for image in query.images]
        logits = self.model.forward_many(flat_images)
        results, offset = [], 0
        differentiable = []
        for query in queries:
            count = len(query.images)
            query_logits = logits[offset:offset + count]
            objectives = self._objectives(query_logits, query)
            results.append(GradientResult(query_logits, objectives))
            if query.variable is not None:
                differentiable.append((len(results) - 1, objectives.sum(), query.variable))
            offset += count
        if differentiable:
            gradients = torch.autograd.grad(
                sum(item[1] for item in differentiable),
                [item[2] for item in differentiable],
            )
            for (result_index, _, _), gradient in zip(differentiable, gradients):
                results[result_index].gradient = gradient
        for result in results:
            result.logits = result.logits.detach()
            result.objectives = result.objectives.detach()
        return results


class AsyncMixin:
    def _init_async(self):
        self._runner = self._pending = self._final_result = None

    @property
    def done(self): return self._final_result is not None

    def ask(self):
        if self.done: return None
        if self._pending is not None: return self._pending
        if self._runner is None: self._runner = self._run_queries()
        try: self._pending = next(self._runner)
        except StopIteration as stop:
            self._final_result = stop.value; return None
        return self._pending

    def tell(self, result):
        if self._pending is None: raise RuntimeError("tell() requires a pending gradient query")
        self._pending = None
        try: self._pending = self._runner.send(result)
        except StopIteration as stop: self._final_result = stop.value

    def result(self): return self._final_result

    def _query(self, image, variable=None, objective="ce"):
        result = yield from self._queries([image], variable, objective)
        return result.logits, result.objectives[0], result.gradient

    def _queries(self, images, variable=None, objective="ce"):
        result = yield GradientQuery(images, self.true_label, self.target_label,
                                     self.targeted, objective, variable)
        for prediction in result.logits.argmax(1).tolist():
            self.query_count += 1
            if self._is_success(int(prediction)) and self.first_success_query is None:
                self.first_success_query = self.query_count
        return result


class AsyncMaskedPGD(AsyncMixin, MaskedPGD):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); self._init_async()
    def _run_queries(self):
        delta = torch.empty_like(self.image_normalized).uniform_(-self.eps, self.eps) * self.mask
        adversarial = (self.image_normalized + delta).clamp(0, 1)
        adversarial = (self.image_normalized * (1-self.mask) + adversarial*self.mask).detach()
        success = False
        for iteration in range(1, self.steps + 1):
            if self._should_update_location(iteration, success): adversarial = self._move_best_patch(True)
            adversarial.requires_grad_(True)
            logits, objective, gradient = yield from self._query(
                self._compose_model_input(adversarial), adversarial, "ce")
            success, _, _ = self._record(iteration, adversarial, logits, objective)
            if (self.early_stop and success) or iteration == self.steps: break
            candidate = adversarial + self.step_size * gradient.sign() * self.mask
            delta = (candidate-self.image_normalized).clamp(-self.eps, self.eps)
            adversarial = (self.image_normalized + delta*self.mask).clamp(0, 1).detach()
        return self._result()


class AsyncMaskedAutoPGD(AsyncMixin, MaskedAutoPGD):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); self._init_async()
    def _run_queries(self):
        delta = torch.empty_like(self.image_normalized).uniform_(-self.eps, self.eps)*self.mask
        adv = (self.image_normalized+delta).clamp(0,1).detach(); previous=adv.clone(); best=adv.clone()
        best_obj=-torch.inf; step=self.step_size; checkpoint=max(self.steps//10,1); history=[]; success=False
        for iteration in range(1,self.steps+1):
            if self._should_update_location(iteration,success):
                adv=self._move_best_patch(True); previous=adv.clone(); best=adv.clone(); best_obj=-torch.inf; step=self.step_size; history=[]
            adv.requires_grad_(True)
            logits,obj,gradient=yield from self._query(self._compose_model_input(adv),adv,"ce")
            success,_,_=self._record(iteration,adv,logits,obj); value=float(obj.item()); history.append(value)
            if value>float(best_obj): best_obj=obj; best=adv.detach().clone()
            if (self.early_stop and success) or iteration==self.steps: break
            projected=adv+step*gradient.sign()*self.mask
            projected=self.image_normalized+(projected-self.image_normalized).clamp(-self.eps,self.eps)*self.mask
            momentum=.75 if iteration>1 else 1.; candidate=adv+momentum*(projected-adv)+(1-momentum)*(adv-previous)
            previous=adv.detach(); adv=(self.image_normalized+(candidate-self.image_normalized).clamp(-self.eps,self.eps)*self.mask).clamp(0,1).detach()
            if iteration%checkpoint==0:
                improvements=sum(b>a for a,b in zip(history,history[1:]))
                if improvements<=len(history)//2: step*=.5; adv=best.clone(); previous=best.clone()
                history=[]
        return self._result()


class AsyncLaVAN(AsyncMixin, LaVAN):
    def __init__(self,*args,**kwargs): super().__init__(*args,**kwargs); self._init_async()
    def _run_queries(self):
        patch=self._initial_patch(); success=False
        adv=self._place_normalized_patch(patch).detach().requires_grad_(True)
        kind="lavan" if self.targeted else "margin"
        _,_,gradient=yield from self._query(self._compose_model_input(adv),adv,kind)
        for iteration in range(1,self.steps+1):
            if self._should_update_location(iteration,success):
                self._set_location(self._sample_location(self.location)); patch=self.best_patch.clone()
                adv=self._place_normalized_patch(patch).detach().requires_grad_(True)
                _,_,gradient=yield from self._query(self._compose_model_input(adv),adv,kind)
            patch=(patch+self.step_size*self._normalized_patch(gradient)).clamp(0,1).detach()
            adv=self._place_normalized_patch(patch).detach().requires_grad_(True)
            logits,obj,gradient=yield from self._query(self._compose_model_input(adv),adv,kind)
            success,_,_=self._record(iteration,adv,logits,obj)
            if self.early_stop and success: break
        return self._result()


class AsyncLOAP(AsyncMixin, LOAP):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); self._init_async()

    def _image_for(self, patch, location):
        row, column = location; ph, pw = self.patch_size
        image = self.image.clone()
        image[:, :, row:row+ph, column:column+pw] = patch * self.value_range + self.clip_min
        return image

    def _run_queries(self):
        global_iteration = 0
        for _ in range(self.attempts):
            self._set_location(self._sample_location())
            patch = self._initial_patch()
            for _ in range(self.steps):
                global_iteration += 1
                patch.requires_grad_(True)
                _, _, gradient = yield from self._query(
                    self._image_for(patch, self.location), patch, "ce")
                patch = (patch + self.step_size * gradient.sign()).clamp(0, 1).detach()

                directions = (tuple(self._DIRECTIONS) if self.lo_mode == "full"
                              else (random.choice(tuple(self._DIRECTIONS)),))
                locations = [self.location]
                for direction in directions:
                    candidate = self._sample_location_candidate(direction)
                    if candidate not in locations: locations.append(candidate)
                best_location, best_value = locations[0], None
                for location in locations:
                    _, objective, _ = yield from self._query(
                        self._image_for(patch, location), objective="ce")
                    value = float(objective.item())
                    if best_value is None or value > best_value:
                        best_location, best_value = location, value
                self._set_location(best_location)

                adversarial = self._place_normalized_patch(patch).detach()
                logits, objective, _ = yield from self._query(
                    self._compose_model_input(adversarial), objective="ce")
                success, _, _ = self._record(global_iteration, adversarial, logits, objective)
                if self.early_stop and success: break
            if self.early_stop and self.best_success: break
        return self._result()

    def _sample_location_candidate(self, direction):
        row_delta, column_delta = self._DIRECTIONS[direction]
        row, column = self.location
        candidate = (row + row_delta * self.stride, column + column_delta * self.stride)
        return candidate if self._location_is_allowed(candidate) else self.location


ASYNC_ATTACKS = {"MaskedPGD": AsyncMaskedPGD, "MaskedAutoPGD": AsyncMaskedAutoPGD,
                 "LaVAN": AsyncLaVAN, "LOAP": AsyncLOAP}
