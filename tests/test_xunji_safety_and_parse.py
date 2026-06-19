import json
import unittest

from xunji.client import XunjiClient, XunjiError, validate_rows
from xunji.muscle_groups import lookup
from xunji.parse import parse_record, parse_response


class ClientResponseClassificationTests(unittest.TestCase):
    def setUp(self):
        self.client = XunjiClient.__new__(XunjiClient)

    def test_fetch_success_false_with_res_list_is_success(self):
        data = {"success": False, "res": ["260601,休息日"]}
        self.assertIs(self.client._classify_fetch_response(data), data)

    def test_upsert_empty_res_list_is_success(self):
        data = {"res": []}
        self.assertIs(self.client._classify_upsert_response(data), data)

    def test_upsert_error_field_wins_over_res_list(self):
        with self.assertRaises(XunjiError):
            self.client._classify_upsert_response({"success": False, "error": "bad row", "res": []})


class UpsertSafetyTests(unittest.TestCase):
    def test_duplicate_rows_rejected(self):
        rows = [
            "2026-06-01,推日,1.卧推,1组,60kg,10次",
            "2026-06-01,推日,1.卧推,1组,60kg,10次",
        ]
        with self.assertRaises(XunjiError):
            validate_rows(rows)

    def test_mixed_dates_rejected(self):
        rows = [
            "2026-06-01,休息日",
            "2026-06-02,休息日",
        ]
        with self.assertRaises(XunjiError):
            validate_rows(rows)

    def test_yyyy_mm_dd_and_yymmdd_accepted(self):
        self.assertEqual(validate_rows(["2026-06-01,休息日"])[0].date, "2026-06-01")
        self.assertEqual(validate_rows(["260601,休息日"])[0].date, "260601")


class ParseTests(unittest.TestCase):
    def test_title_and_iso_date_parsed(self):
        rec = parse_record("2026-06-01,id:abc,推日,1.杠铃卧推,1组,60kg,10次")
        self.assertEqual(rec["date"], "2026-06-01")
        self.assertEqual(rec["date_code"], "260601")
        self.assertEqual(rec["title"], "推日")
        self.assertEqual(rec["local_id"], "abc")

    def test_short_date_parsed_to_iso(self):
        rec = parse_record("260601,id:abc,拉日,1.杠铃划船,1组,75kg,7次")
        self.assertEqual(rec["date"], "2026-06-01")
        self.assertEqual(rec["date_code"], "260601")

    def test_cardio_metrics_parsed(self):
        rec = parse_record("260604,有氧,1.划船机,4.72km,228kcal,138bpm,time:3889s,2.爬楼梯,floors:91,378kcal,time:1800s,158bpm")
        self.assertEqual(rec["exercises"][0]["modality"], "cardio")
        self.assertEqual(rec["exercises"][0]["cardio"]["distance_km"], 4.72)
        self.assertEqual(rec["exercises"][0]["cardio"]["kcal"], 228)
        self.assertEqual(rec["exercises"][1]["cardio"]["floors"], 91)
        self.assertEqual(rec["total_sets"], 0)

    def test_assisted_pullup_load_kind(self):
        rec = parse_record("260601,拉日,1.引体向上（辅助）,1组,10kg,8次")
        ex = rec["exercises"][0]
        st = ex["sets"][0]
        self.assertEqual(ex["load_kind"], "assist_kg")
        self.assertEqual(st["assist_kg"], 10.0)


class MuscleGroupTests(unittest.TestCase):
    def test_recent_real_exercises_are_known(self):
        names = [
            "杠铃深蹲", "站姿提踵", "杠铃罗马尼亚硬拉", "爬楼梯",
            "划船机", "坐姿器械卷腹", "肩袖热身-Pallof Press", "Machine Fly",
        ]
        for name in names:
            with self.subTest(name=name):
                primary, _ = lookup(name)
                self.assertNotEqual(primary, ["unknown"])


if __name__ == "__main__":
    unittest.main()
