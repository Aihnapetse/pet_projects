from bs4 import BeautifulSoup
import re

def parse_product_info(html: str) -> dict:
    """
    Extracts product information from the given HTML.

    Parameters:
        html (str): The input HTML data.

    Returns:
        product_info (dict): A dictionary containing the product's info.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # title: product-title OR <h1>
    title_tag = soup.find(class_='product-title') or soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else None

    # category: product-category OR <ul class="breadcrumb">
    category_tag = soup.find(class_='product-category')
    if not category_tag:
        crumbs = soup.select('ul.breadcrumb li a')
        if crumbs:
            category_tag = crumbs[-1]
    category = category_tag.get_text(strip=True) if category_tag else None

    # old_price: old-price OR <del>
    old_price_tag = soup.find(class_='old-price') or soup.find('del')
    old_price = old_price_tag.get_text(strip=True) if old_price_tag else None

    # new_price: new-price OR <ins>, OR old_price_tag
    new_price_tag = soup.find(class_='new-price') or soup.find('ins')
    if not new_price_tag and old_price_tag:
        sibling = old_price_tag.find_next(string=re.compile(r'\d'))
        if sibling:
            new_price = sibling.strip()
        else:
            new_price = None
    else:
        new_price = new_price_tag.get_text(strip=True) if new_price_tag else None

    product_info = {
        'title': title,
        'category': category,
        'old_price': old_price,
        'new_price': new_price
    }

    return product_info
