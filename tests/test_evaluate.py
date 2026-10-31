import copy
import unittest

from scripts.evaluate import score


def clip(clip_id, reference, **extra):
    return dict(id=clip_id, speaker_id="person", session_id="day4", split="test",
                task="dictation", mode="silent", reference=reference,
                duration_s=2.0, phrase_family=clip_id) | extra


class EvaluationContractTest(unittest.TestCase):
    def test_corpus_errors_and_complete_denominator(self):
        rows = [clip("a", "Send the file."), clip("b", "Do not send it")]
        predictions = [{"id": "a", "text": "send a file", "processing_s": 1},
                       {"id": "b", "text": "", "processing_s": 2}]
        metrics = score(rows, predictions, "test")["groups"]["dictation/silent"]
        self.assertAlmostEqual(metrics["wer"], 5 / 7)
        self.assertEqual(metrics["deletions"], 4)
        self.assertEqual(metrics["substitutions"], 1)
        self.assertEqual(metrics["processing_rtf"], 0.75)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            score(rows, predictions[:1], "test")

    def test_session_and_phrase_leakage(self):
        test = clip("test", "Send it tomorrow")
        train = clip("train", "Call me later", split="train")
        with self.assertRaisesRegex(ValueError, "Session crosses"):
            score([train, test], [{"id": "test", "text": ""}], "test")
        train["session_id"] = "day1"
        for field, value in [("reference", "send it tomorrow!"), ("phrase_family", "test")]:
            leaking = copy.deepcopy(train)
            leaking[field] = value
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                score([leaking, test], [{"id": "test", "text": ""}], "test")
        train.update(task="command", label="send", reference=test["reference"])
        with self.assertRaisesRegex(ValueError, "Novel phrase crosses"):
            score([train, test], [{"id": "test", "text": ""}], "test")
        test.update(task="command", label="__unknown__")
        with self.assertRaisesRegex(ValueError, "Conflicting command labels"):
            score([train, test], [{"id": "test", "text": "", "label": "send"}], "test")
        test["label"] = "send"
        score([train, test], [{"id": "test", "text": "", "label": "send"}], "test")

    def test_rejection_and_speech_modes_remain_separate(self):
        rows = [clip("a", "send it", task="command", label="send"),
                clip("b", "stop it", task="command", label="stop"),
                clip("c", "paint it", task="command", label="__unknown__"),
                clip("d", "", task="idle", mode="non_speech", label="__unknown__"),
                clip("e", "send it", mode="voiced_muted")]
        predictions = [{"id": r["id"], "text": "", "label": label} for r, label in
                       zip(rows, ["send", "__unknown__", "send", "__unknown__", "unused"])]
        groups = score(rows, predictions, "test")["groups"]
        command = groups["command/silent"]
        self.assertEqual(command["known_coverage"], 0.5)
        self.assertEqual(command["accepted_known_accuracy"], 1.0)
        self.assertEqual(command["false_accept_rate"], 1.0)
        self.assertEqual(groups["idle/non_speech"]["false_accepts"], 0)
        self.assertIn("dictation/voiced_muted", groups)


if __name__ == "__main__":
    unittest.main()
