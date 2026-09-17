from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from road2ai_vifinqa import reasoning_llm


class ReasoningClientTest(unittest.TestCase):
    def invoke(self, message, finish="stop"):
        data = {"choices": [{"message": message, "finish_reason": finish}], "usage": {}}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(data).encode())) as mock:
            result = reasoning_llm.chat(system="Audit", user="Question", model="local", base_url="http://127.0.0.1:1")
            payload = json.loads(mock.call_args.args[0].data)
        return result, payload

    def test_final_content_is_separate(self):
        result, payload = self.invoke({"content": '{"ok": true}', "reasoning_content": '{"ok": false}'})
        self.assertEqual(result.content, '{"ok": true}')
        self.assertEqual(payload["temperature"], 0.6)
        self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("/no_think", json.dumps(payload))

    def test_reasoning_only_rejected(self):
        with self.assertRaises(reasoning_llm.IncompleteCompletion):
            self.invoke({"content": None, "reasoning_content": '{"answer": 123}'})

    def test_truncated_valid_json_rejected(self):
        with self.assertRaises(reasoning_llm.IncompleteCompletion):
            self.invoke({"content": '{"answer": 123}'}, "length")

    def test_tagged_reasoning_stripped_before_json(self):
        result, _ = self.invoke({"content": '<think>{"wrong": 1}</think>\n{"right": 2}'})
        self.assertEqual(result.content, '{"right": 2}')

    def test_unclosed_reasoning_rejected(self):
        with self.assertRaises(reasoning_llm.IncompleteCompletion):
            self.invoke({"content": '<think>{"answer": 1}'})


if __name__ == "__main__":
    unittest.main()
