import unittest
from unittest.mock import patch, sentinel

from road2ai_vifinqa.hard_note_solver import solve_note


class NotePanelInjectionTest(unittest.TestCase):
    def test_caller_panel_reaches_deterministic_scenario(self):
        with patch("road2ai_vifinqa.hard_note_solver._deterministic_solution", return_value=sentinel.result) as solve:
            result = solve_note("scenario", 430, sentinel.corpus, panel=sentinel.panel)
        self.assertIs(result, sentinel.result)
        self.assertIs(solve.call_args.kwargs["panel"], sentinel.panel)


if __name__ == "__main__":
    unittest.main()
