"""
keepalive.py
Actually visits the streamlit app with a real headless browser, since a plain
curl/HTTP GET only returns a static "Zzz" HTML page and never launches the real
app. If the sleep screen appears, this clicks the wake-up button and waits for
the real app to load. If the app is already awake, it does nothing and exits cleanly.
"""

from playwright.sync_api import sync_playwright
import sys

APP_URL = "https://breast-cancer-classifier-9jvbxy7lxwanxqcw5o5xnc.streamlit.app/"
WAKE_BUTTON_TEXT = "Yes, get this app back up!"

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        try:
            print(f"visiting {APP_URL}")
            page.goto(APP_URL, timeout=30000, wait_until="domcontentloaded")

            # give the page a moment to render the sleep screen if it's going to
            page.wait_for_timeout(3000)

            wake_button = page.get_by_text(WAKE_BUTTON_TEXT)

            if wake_button.count() > 0:
                print("app is asleep, clicking wake-up button")
                wake_button.first.click()
                # streamlit apps take a while to actually boot back up
                page.wait_for_timeout(20000)
                print("clicked wake-up, waited for app to boot")
            else:
                print("no sleep screen found, app is already awake")

        except Exception as e:
            print(f"error during check: {e}")
            browser.close()
            sys.exit(1)

        browser.close()
        print("done")

if __name__ == "__main__":
    main()
