// methods ページのスクリプトは数式描画 (KaTeX) のみ。
// 以前あったスクロール連動の演出・スクロールスパイ・スムーズスクロール等の
// インタラクションは廃止し、内容は常時静的に表示する。
document.addEventListener("DOMContentLoaded", function () {
  if (typeof renderMathInElement !== "function") return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false }
    ],
    throwOnError: false,
    errorColor: "#dc2626",
    strict: "ignore"
  });
});
