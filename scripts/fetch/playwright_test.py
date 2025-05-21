import playwright
from playwright.sync_api import sync_playwright

def run(playwright):
    iphone_13 = playwright.devices['iPhone 13']
    browser = playwright.webkit.launch(headless=False)
    context = browser.new_context(
        **iphone_13,
    )
    return context

def main():
    with sync_playwright() as p:
        iphone_13 = p.devices['iPhone 13']
        browser = p.webkit.launch(headless=False)
        context = browser.new_context(
            **iphone_13,
        )
        page = context.new_page()
        page.goto("https://www.tiktok.com/@therock/video/7341471780412801024")
        print(page.title())
        browser.close()

if __name__ == '__main__':
    main()