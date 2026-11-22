# -*- coding: utf-8 -*-
"""g007 KakaoTalk notifier: the picks message (code + name only) split to the template limit, is_configured from the two files,
and send_text / send_picks answering False instead of ending the caller when the setup is missing."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_simulation"))

import g007_kakao_notify as g007  # noqa: E402


class PicksMessageTest(unittest.TestCase):
    def test_code_and_name_only(self):
        msgs = g007.picks_messages(["204620_글로벌텍스프리", "323410,카카오뱅크", "008930 한미사이언스"], "20261120")
        self.assertEqual(msgs, ["[20261120 선별 3종목]\n204620 글로벌텍스프리\n323410 카카오뱅크\n008930 한미사이언스"])

    def test_no_picks(self):
        self.assertEqual(g007.picks_messages([], "20261120"), ["[20261120 선별] 선정 종목 없음"])

    def test_split_to_the_limit_and_nothing_lost(self):
        picks = [f"{100000 + k}_종목이름이조금긴회사{k}" for k in range(40)]
        msgs = g007.picks_messages(picks, "20261120")
        self.assertGreater(len(msgs), 1)
        for k, m in enumerate(msgs, 1):
            self.assertLessEqual(len(m), g007.TEXT_TEMPLATE_MAX_CHARS)
            self.assertTrue(m.startswith(f"[20261120 선별 40종목] ({k}/{len(msgs)})"))
        lines = [line for m in msgs for line in m.split("\n")[1:]]
        self.assertEqual(lines, [f"{100000 + k} 종목이름이조금긴회사{k}" for k in range(40)])

    def test_name_with_underscore_keeps_its_rest(self):
        self.assertEqual(g007.picks_messages(["950260_인제니아_테라퓨틱스"], "L")[0].split("\n")[1], "950260 인제니아_테라퓨틱스")


class SendTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.patches = [mock.patch.object(g007, "CONFIG_DIR", d), mock.patch.object(g007, "CONFIG_PATH", d / "kakao_config.json"),
                        mock.patch.object(g007, "TOKEN_PATH", d / "kakao_token.json")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_not_configured_is_false_not_an_exit(self):
        self.assertFalse(g007.is_configured())
        self.assertFalse(g007.send_text("hello"))            # SystemExit of the missing config must not reach the caller
        self.assertFalse(g007.send_picks(["204620_글로벌텍스프리"], "20261120"))

    def test_configured_needs_both_files(self):
        g007.CONFIG_PATH.write_text('{"rest_api_key": "k", "redirect_uri": "http://localhost:5000/oauth"}', encoding="utf-8")
        self.assertFalse(g007.is_configured())
        g007.TOKEN_PATH.write_text("{}", encoding="utf-8")
        self.assertTrue(g007.is_configured())

    def test_send_picks_sends_every_part_and_stops_on_failure(self):
        picks = [f"{100000 + k}_종목이름이조금긴회사{k}" for k in range(40)]
        parts = g007.picks_messages(picks, "20261120")
        with mock.patch.object(g007, "send_text", return_value=True) as st:
            self.assertTrue(g007.send_picks(picks, "20261120"))
            self.assertEqual([c.args[0] for c in st.call_args_list], parts)
        with mock.patch.object(g007, "send_text", side_effect=[True, False, True]) as st:
            self.assertFalse(g007.send_picks(picks, "20261120"))
            self.assertEqual(st.call_count, 2)


if __name__ == "__main__":
    unittest.main()
