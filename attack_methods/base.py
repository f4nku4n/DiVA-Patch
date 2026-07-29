from abc import abstractmethod


class Attacker:
    def __init__(self, max_query):
        self.process = []

        self.n_query = 0
        self.max_query = max_query
        self.first_success_query = None

    def count_query(self, success):
        self.n_query += 1
        if success and self.first_success_query is None:
            self.first_success_query = self.n_query

    @abstractmethod
    def run(self, **kwargs):
        pass
