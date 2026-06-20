import importlib
import os
import sys
import unittest


class SummaryAgentImportTest(unittest.TestCase):
    def test_api_mode_import_does_not_eagerly_load_local_model_dependencies(self):
        os.environ["USE_LOCAL_MODEL"] = "api"
        sys.modules.pop("src.agents.summary_agent", None)
        sys.modules.pop("torch", None)
        sys.modules.pop("transformers", None)

        summary_agent = importlib.import_module("src.agents.summary_agent")

        self.assertEqual(summary_agent.get_model_choice(), "api")
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)


if __name__ == "__main__":
    unittest.main()
