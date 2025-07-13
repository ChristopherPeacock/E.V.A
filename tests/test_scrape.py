from utils.scrape import scrape_url


def test_scrape_url_returns_scraped_data():
    result = scrape_url('https://www.opticompliance.co.uk/')
    assert 'fire compliance software' in result.lower()

