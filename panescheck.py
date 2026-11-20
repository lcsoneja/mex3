import sys

from playwright.sync_api import sync_playwright

P = []
def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok: P.append(msg)

with sync_playwright() as p:
    b = p.chromium.launch(channel='msedge')
    pg = b.new_page(viewport={'width': 1680, 'height': 1000})
    errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto('http://127.0.0.1:8765/', wait_until='load')
    pg.wait_for_timeout(3500)

    # split bubbles: user right, agent left (the engine restarts empty, so send a real turn)
    pg.click('.rail button[data-view=\'chat\']'); pg.wait_for_timeout(600)
    pg.fill('#input', 'Reply with exactly: panes ok')
    pg.keyboard.press('Enter')
    pg.wait_for_timeout(12000)
    check(pg.locator('#transcript .trow.user').count() >= 1, 'user bubbles present')
    check(pg.locator('#transcript .trow.agent').count() >= 1, 'agent bubbles present (split)')
    check(pg.locator('#transcript .latchip').count() >= 1, 'latency chips on replies')

    # six panes
    for pane in ('pipeline','trace','timeline','agents','workflow','mindmap'):
        check(pg.locator('#pane-' + pane).count() == 1, 'pane exists: ' + pane)
    check(pg.locator('#tab-pipeline .pv').count() == 8, 'pipeline shows the 8 stages mic-to-speaker')
    check(pg.locator('#tab-agents .agedit').count() >= 6, 'agents pane lists roster (>=6 roles)')
    check('doctor' in pg.inner_text('#tab-agents'), 'doctor visible in agents pane')
    check(pg.locator('#tab-agents select').count() >= 12, 'model + thinking selects in agents pane')
    check(pg.locator('#tab-workflow.tree').count() == 1, 'workflow tree renders')
    check('└' in pg.inner_text('#tab-workflow'), 'workflow shows hierarchical tree')

    # wake toggle at top of conversation
    check(pg.locator('#btn-wake').count() == 1, 'wake toggle in topbar')
    txt = pg.inner_text('#btn-wake')
    check('wake' in txt.lower(), 'wake label renders (' + txt.strip() + ')')

    # rail collapse + lock
    pg.click('#btn-rail'); pg.wait_for_timeout(400)
    cls = pg.get_attribute('.rail', 'class')
    check('collapsed' in (cls or ''), 'rail collapses')
    pg.click('#btn-rail'); pg.wait_for_timeout(300)
    cls = pg.get_attribute('.rail', 'class')
    check('collapsed' not in (cls or ''), 'rail expands again')
    pg.click('#btn-rail-lock'); pg.wait_for_timeout(200)
    pg.click('#btn-rail'); pg.wait_for_timeout(300)
    cls = pg.get_attribute('.rail', 'class')
    check('collapsed' not in (cls or ''), 'locked rail refuses to collapse')
    pg.click('#btn-rail-lock'); pg.wait_for_timeout(200)

    # divider drag: shrink inspector then restore
    before = pg.evaluate("() => getComputedStyle(document.documentElement).getPropertyValue('--inspw')")
    box = pg.locator('#divider').bounding_box()
    if box:
        pg.mouse.move(box['x'] + 3, box['y'] + 300)
        pg.mouse.down(); pg.mouse.move(box['x'] - 120, box['y'] + 300, steps=4); pg.mouse.up()
        pg.wait_for_timeout(300)
        after = pg.evaluate("() => getComputedStyle(document.documentElement).getPropertyValue('--inspw')")
        check(before != after, 'divider drag resizes panes (' + str(after)[:8] + 'px)')
    else:
        check(False, 'divider not visible to drag')

    first_err = errs[0][:120] if errs else ''
    check(not errs, 'no page errors: ' + first_err)
    pg.screenshot(path='out/ui_v2.png')
    b.close()

print()
print('RESULT: ' + ('FAIL' if P else 'PASS'))
for x in P: print('PROBLEM:', x)
sys.exit(1 if P else 0)