class MAP_Elites:
    def __init__(self, img_h, img_w, cell_h, cell_w):
        self.cell_h = cell_h
        self.cell_w = cell_w
        self.grid_search = None
        self.grid_final = None
        self.n_visited = None

        self.list_index = None
        self.range_w = None
        self.range_h = None
        self.max_h, self.max_w = None, None
        self.init(img_h, img_w)

        self.idx_best_search = None
        self.idx_best_final = None

        self.activated_cells = []

    def init(self, img_h, img_w):
        self.grid_search = {}
        self.grid_final = {}

        self.n_visited = {}
        range_x = (img_h // self.cell_h) - 1 if img_h % self.cell_h == 0 else img_h // self.cell_h
        range_y = (img_w // self.cell_w) - 1 if img_w % self.cell_w == 0 else img_w // self.cell_w

        for x in range(range_x + 1):
            for y in range(range_y + 1):
                self.grid_search[f"{x}-{y}"] = None
                self.grid_final[f"{x}-{y}"] = None
                self.n_visited[f"{x}-{y}"] = 0

        self.max_h = range_x
        self.max_w = range_y
        self.list_index = list(self.grid_search.keys())

    def get_index(self, idv):
        x, y = idv.location
        if x == 0:
            i = 0
        else:
            i = (x // self.cell_h) - 1 if x % self.cell_h == 0 else x // self.cell_h
        if y == 0:
            j = 0
        else:
            j = (y // self.cell_w) - 1 if y % self.cell_w == 0 else y // self.cell_w
        return i, j

    def assign(self, idv, it=None):
        i, j = self.get_index(idv)
        idx = f'{i}-{j}'
        self.n_visited[idx] += 1

        if self.grid_search[idx] is None:
            self.grid_search[idx] = idv
            self.grid_final[idx] = idv
            self.activated_cells.append(idx)
        else:
            if self.is_strictly_better(new_idv=idv, cur_idv=self.grid_search[idx]):
                self.grid_search[idx] = idv

            if self.is_strictly_better(new_idv=idv, cur_idv=self.grid_final[idx]):
                self.grid_final[idx] = idv

        if self.idx_best_search is None or self.is_strictly_better(idv, self.grid_search[self.idx_best_search]):
            self.idx_best_search = idx

        if self.idx_best_final is None or self.is_strictly_better(idv, self.grid_final[self.idx_best_final]):
            self.idx_best_final = idx

    @staticmethod
    def is_strictly_better(new_idv, cur_idv):
        if not cur_idv.success_attack and new_idv.success_attack:
            return True
        elif cur_idv.success_attack and new_idv.success_attack:
            return new_idv.l2 < cur_idv.l2
            # return new_idv.ssim_patch > cur_idv.ssim_patch
        elif not cur_idv.success_attack and not new_idv.success_attack:
            return new_idv.loss < cur_idv.loss
        return False
