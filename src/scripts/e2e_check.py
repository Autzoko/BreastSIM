#!/usr/bin/env python
"""End-to-end check of the Probe Lab UI in headless Chromium (Playwright).

Drives the running server like a user: loads a phantom, jumps to a lesion,
cycles the scan presets, moves the probe with the keyboard and by clicking
the 3D skin, and fails on any browser console / page error.

    PLAYWRIGHT_BROWSERS_PATH=... python src/scripts/e2e_check.py --url http://127.0.0.1:8765
"""
import argparse
import json
import sys
import time

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--phantom", default="usct/A")
    ap.add_argument("--out", default="outputs/e2e.png")
    ap.add_argument("--timeout", type=float, default=600, help="s, includes a possible cache build")
    args = ap.parse_args()
    errors, report = [], {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("requestfailed", lambda r: errors.append(f"requestfailed: {r.url}"))
        page.goto(args.url)
        page.wait_for_function("() => window.__breastsim && window.__breastsim.config", timeout=30_000)

        def frame_after(seq, timeout=60_000):
            page.wait_for_function(f"() => (window.__breastsim.lastMeta||{{}}).seq > {seq}", timeout=timeout)
            return page.evaluate("() => window.__breastsim.lastMeta")

        page.select_option("#phantom", args.phantom)
        page.wait_for_function(
            f"() => window.__breastsim.info && window.__breastsim.info.id === '{args.phantom}' && window.__breastsim.lastMeta",
            timeout=args.timeout * 1000)
        report["lesions_listed"] = page.evaluate("() => document.querySelectorAll('#lesions .lesion').length")
        report["meshes_3d"] = page.evaluate("() => window.__breastsim.meshCount")

        # jump to the first lesion: it must appear in the image plane
        seq = page.evaluate("() => window.__breastsim.seq")
        page.click("#lesions button")
        meta = frame_after(seq)
        report["go_lesion_area_mm2"] = meta["lesion_area_mm2"]
        if not meta["lesion_area_mm2"] > 0:
            errors.append("lesion not in plane after 'Go'")

        # every scan preset renders at its pixel spacing
        presets = page.evaluate("() => window.__breastsim.config.presets")
        report["presets"] = {}
        for name, pr in presets.items():
            seq = page.evaluate("() => window.__breastsim.seq")
            page.select_option("#preset", name)
            meta = frame_after(seq)
            ok = abs(meta["px_x_mm"] - pr["px_x_mm"]) < 0.01 and abs(meta["px_z_mm"] - pr["px_z_mm"]) < 0.01
            report["presets"][name] = {"render_ms": meta["render_ms"], "px": [round(meta["px_x_mm"], 3), round(meta["px_z_mm"], 3)], "ok": ok}
            if not ok:
                errors.append(f"preset {name}: pixel spacing {meta['px_x_mm']}, {meta['px_z_mm']}")
        page.select_option("#preset", "breast")

        # keyboard moves the probe along the array
        u0 = page.evaluate("() => window.__breastsim.pose.u")
        page.locator("body").click(position={"x": 5, "y": 5})
        for _ in range(4):
            page.keyboard.press("d")
        u1 = page.evaluate("() => window.__breastsim.pose.u")
        report["keyboard_du_mm"] = round(u1 - u0, 2)
        if abs((u1 - u0) - 2.0) > 1e-6:
            errors.append(f"keyboard move: du={u1 - u0}")

        # clicking the 3D skin places the probe
        pose0 = page.evaluate("() => ({...window.__breastsim.pose})")
        box = page.locator("#scene3d canvas").bounding_box()
        page.mouse.click(box["x"] + box["width"] * 0.47, box["y"] + box["height"] * 0.42)
        time.sleep(0.5)
        pose1 = page.evaluate("() => ({...window.__breastsim.pose})")
        report["click_moved_probe"] = (pose0["u"], pose0["v"]) != (pose1["u"], pose1["v"])

        time.sleep(1.5)
        report["fps_readout"] = page.inner_text("#perf").split(" fps")[0]
        page.screenshot(path=args.out)
        browser.close()

    report["errors"] = errors
    print(json.dumps(report, indent=1))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
