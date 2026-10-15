//! Shell styling — one embedded stylesheet (no asset pipeline needed for
//! the CSR build).

pub const CSS: &str = r#"
:root { color-scheme: dark; }
* { box-sizing: border-box; }

.mode-switch { display: flex; gap: 6px; margin-bottom: 2px; }
.mode-switch button { background: #202836; color: #9fb0c3; border: 1px solid #2c3a4e; border-radius: 6px; padding: 4px 12px; cursor: pointer; }
.mode-switch button.active { background: #2b425c; color: #e8f1fa; border-color: #3f6a96; }
.studio-grid { display: grid; grid-template-columns: 260px minmax(480px, 1fr) 300px; gap: 12px; align-items: start; }
.app { max-width: none; }
.grid-left, .grid-right { display: flex; flex-direction: column; gap: 10px; }
.grid-center { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
@media (max-width: 1100px) { .studio-grid { grid-template-columns: 1fr; } }
/* GEN-57 workspace tabs (Zemax-style document tab strip) */
.wtab-bar { display: flex; gap: 2px; flex-wrap: wrap; border-bottom: 1px solid #232c38; padding: 0 2px; }
.wtab { background: transparent; color: #7d8ea3; border: 1px solid transparent; border-bottom: none; border-radius: 6px 6px 0 0; padding: 5px 12px; cursor: pointer; font-size: 12.5px; }
.wtab:hover { color: #cfe0f2; }
.wtab.active { background: #161c24; color: #e8f1fa; border-color: #232c38; }
.workspace { display: flex; flex-direction: column; gap: 10px; }
/* GEN-57 System Explorer tree */
.tree { padding: 8px 10px; }
.tree-group { margin-bottom: 8px; }
.tree-h { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.8px; color: #5f7086; margin: 6px 0 3px; }
.tree-item { display: block; width: 100%; text-align: left; background: transparent; border: 0; border-left: 2px solid transparent; color: #9fb0c3; padding: 3px 8px 3px 12px; cursor: pointer; font-size: 12.5px; border-radius: 0 5px 5px 0; }
.tree-item:hover { background: #1b2230; color: #dbe2ea; }
.tree-item.active { background: #20293a; color: #e8f1fa; border-left-color: #4b8fd1; }
/* GEN-57 Lens Data toolbar + selection */
.editor-toolbar { display: flex; gap: 5px; flex-wrap: wrap; align-items: center; margin-bottom: 8px; }
.editor-toolbar .btn { padding: 3px 9px; font-size: 12px; }
.editor-toolbar .btn.danger { background: #4a2431; border-color: #6b2435; }
.editor-toolbar .btn.apply { background: #1d4a34; border-color: #2f6b4c; }
table.surf tr.selected td { background: #24405c; }
table.surf tbody tr { cursor: pointer; }
/* GEN-57 summary card */
.sum-row { display: flex; justify-content: space-between; gap: 10px; font-size: 12px; color: #7d8ea3; padding: 1.5px 0; }
.sum-row .val { color: #d7dce3; font-family: ui-monospace, monospace; text-align: right; }
/* GEN-57 live layout panel inside the Lens Data Editor */
.live-layout-row { display: flex; gap: 14px; align-items: flex-start; margin-top: 6px; }
.psf-img.mini { width: 240px; max-width: 45%; flex: none; }
.banner.mini { padding: 4px 8px; margin-top: 4px; }
.panel { background: #161c24; border: 1px solid #232c38; border-radius: 8px; padding: 10px 12px; }
.panel h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #7d8ea3; margin: 0 0 8px; }
button { font: inherit; }
.btn { background: #2b425c; color: #e8f1fa; border: 1px solid #3f6a96; border-radius: 6px; padding: 4px 10px; cursor: pointer; }
.btn:disabled { opacity: 0.45; cursor: default; }
input, select, textarea { background: #0e1218; color: #d7dce3; border: 1px solid #2c3a4e; border-radius: 5px; padding: 3px 6px; font: inherit; }
textarea.editor { width: 100%; min-height: 180px; font-family: ui-monospace, monospace; font-size: 12px; resize: vertical; }
.row { display: flex; align-items: center; gap: 6px; margin: 3px 0; flex-wrap: wrap; }
.row label { color: #9fb0c3; }
.grid2 { display: grid; grid-template-columns: auto 1fr; gap: 4px 8px; align-items: center; }
.num { width: 86px; }
.small { font-size: 11px; color: #7d8ea3; }
.err { color: #ff8f8f; white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 12px; }
.warn { color: #ffd479; }
.badge { display: inline-block; border-radius: 4px; padding: 1px 6px; font-size: 11px; }
.badge.edited { background: #4d3b1c; color: #ffd479; }
.badge.stale { background: #55222b; color: #ff9db0; }
table.surf { border-collapse: collapse; width: 100%; }
table.surf th { font-size: 10px; text-transform: uppercase; color: #6c7f95; text-align: left; padding: 2px 4px; }
table.surf td { padding: 2px 4px; border-top: 1px solid #202937; vertical-align: top; }
table.surf tr.flagged td { background: #33260f; }
table.surf input[type=number] { width: 74px; }
table.surf input[type=range] { width: 92px; vertical-align: middle; }
.stop-flag { color: #7fd1ff; font-weight: 600; }
.view-wrap { position: relative; background: #0b0e13; border: 1px solid #232c38; border-radius: 8px; padding: 8px; display: flex; flex-direction: column; gap: 6px; }
.psf-img { width: 100%; image-rendering: pixelated; border-radius: 4px; background: #000; }
.strip-svg { width: 100%; background: #0b0e13; }
.banner { border-radius: 6px; padding: 6px 10px; font-family: ui-monospace, monospace; font-size: 11.5px; white-space: pre-wrap; }
.banner.banner-ok { background: #10291b; color: #7fd6a0; border: 1px solid #1e4a2f; }
.banner.banner-mitigated { background: #33260f; color: #ffd479; border: 1px solid #5c4315; }
.banner.banner-rejected { background: #3a1420; color: #ff9db0; border: 1px solid #6b2435; }
.banner.banner-advisory { background: #14273a; color: #8fc7ff; border: 1px solid #274a6b; }
.readouts { display: flex; gap: 14px; flex-wrap: wrap; color: #9fb0c3; font-family: ui-monospace, monospace; font-size: 11.5px; }
.toasts { position: fixed; right: 14px; bottom: 14px; display: flex; flex-direction: column; gap: 6px; max-width: 460px; z-index: 10; }
.toast { background: #3a1420; color: #ffc7d1; border: 1px solid #6b2435; border-radius: 8px; padding: 8px 10px; display: flex; gap: 8px; align-items: flex-start; box-shadow: 0 6px 18px rgba(0,0,0,0.4); }
.toast.info { background: #142c3a; color: #bfe3ff; border-color: #275a7a; }
.toast button { background: transparent; border: none; color: inherit; cursor: pointer; }
.console { font-family: ui-monospace, monospace; font-size: 11.5px; color: #9fb0c3; white-space: pre-wrap; }
.slider-row { display: grid; grid-template-columns: 130px 1fr 90px; gap: 8px; align-items: center; }
.slider-row output { font-family: ui-monospace, monospace; text-align: right; }
.recent-list { display: flex; flex-wrap: wrap; gap: 4px; }
.recent-list button { background: #202836; border: 1px solid #2c3a4e; color: #9fb0c3; border-radius: 4px; padding: 1px 7px; cursor: pointer; }
.wave-row { display: flex; align-items: center; gap: 8px; margin: 2px 0; }
"#;
