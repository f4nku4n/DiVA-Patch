from abc import abstractmethod


class Attacker:
    def __init__(self, max_query):
        self.process = []

        self.n_query = 0
        self.max_query = max_query

    @abstractmethod
    def run(self, **kwargs):
        pass