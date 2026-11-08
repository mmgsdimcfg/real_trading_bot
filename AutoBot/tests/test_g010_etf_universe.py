"""g010 ETF 추가: 대상/제외 판정, 누락 ETF 전부 추가, 위험 주석, 수동 '#' 존중, --no-etf."""
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_simulation"))
import g010_universe_update as g  # noqa: E402


def rec(code, name, grp="EF", listed="20200101", **kw):
    r = {"code": code, "name": name, "grp": grp, "listed": listed, "market": "KOSPI", "spac": "N",
         "short_over": "0", "halt": "N", "liquidation": "N", "admin": "N", "warn": "00", "warn_pre": "N",
         "unfaithful": "N", "short_hot": "", "runup": "N"}
    r.update(kw)
    return r


class G010EtfTest(unittest.TestCase):
    def setUp(self):
        self.today = datetime(2026, 11, 4)
        self.master = {r["code"]: r for r in [
            rec("000020", "동화약품", grp="ST", listed="19760324"),
            rec("069500", "KODEX 200"),
            rec("122630", "KODEX 레버리지"),
            rec("114800", "KODEX 인버스"),
            rec("148070", "KOSEF 국고채10년"),
            rec("449170", "TIGER 미국테크TOP10타겟커버드콜"),
            rec("357870", "TIGER CD금리투자KIS(합성)"),
            rec("261920", "ACE MSCI필리핀(합성)", halt="Y"),
            rec("0255G0", "마이티 신규ETF", listed="20261103"),
            rec("091160", "KODEX 반도체"),
        ]}

    def _lines(self, extra=()):
        return [g.Line("header", "code,name,market"), g.Line("active", "", "000020", "동화약품", "KOSPI"), *extra]

    def test_target_filter(self):
        m = self.master
        for c in ("069500", "122630", "114800", "091160"):
            self.assertTrue(g.is_target_etf(m[c]), c)
        for c in ("148070", "449170", "357870", "000020"):
            self.assertFalse(g.is_target_etf(m[c]), c)

    def test_adds_all_missing_targets_with_risk_comment(self):
        lines = self._lines()
        g.update_universe(lines, self.master, self.today, None)
        by = {ln.code: ln for ln in lines if ln.code}
        self.assertEqual(set(by) - {"000020"}, {"069500", "122630", "114800", "091160", "261920", "0255G0"})
        self.assertEqual(by["261920"].kind, "auto")
        self.assertIn("거래정지", by["261920"].reasons)
        self.assertEqual(by["0255G0"].kind, "auto")
        self.assertTrue(any(r.startswith("신규상장") for r in by["0255G0"].reasons))
        self.assertEqual(by["069500"].kind, "active")
        self.assertEqual(by["069500"].market, "KOSPI")

    def test_manual_comment_respected_and_idempotent(self):
        lines = self._lines([g.Line("manual", "# 122630,KODEX 레버리지,KOSPI", "122630", "KODEX 레버리지", "KOSPI")])
        g.update_universe(lines, self.master, self.today, None)
        self.assertEqual([ln.kind for ln in lines if ln.code == "122630"], ["manual"])
        rep = g.update_universe(lines, self.master, self.today, None)
        self.assertEqual(rep["added"], [])

    def test_hand_added_new_etf_does_not_block_older_ones(self):
        lines = self._lines([g.Line("active", "", "0255G0", "마이티 신규ETF", "KOSPI")])
        g.update_universe(lines, self.master, self.today, None)
        self.assertIn("069500", {ln.code for ln in lines})

    def test_no_etf(self):
        lines = self._lines()
        rep = g.update_universe(lines, self.master, self.today, None, include_etf=False)
        self.assertEqual(rep["added"], [])


if __name__ == "__main__":
    unittest.main()
