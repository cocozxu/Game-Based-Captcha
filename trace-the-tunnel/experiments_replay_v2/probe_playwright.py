"""5-line probe: confirm playwright's page.mouse.move API works."""
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("about:blank")
        await page.mouse.move(100, 100)
        await page.mouse.down(button="left")
        await page.mouse.move(150, 150)
        await page.mouse.up(button="left")
        print("OK: page.mouse.{move,down,up} all callable")
        await browser.close()

asyncio.run(main())
