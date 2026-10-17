"""Fake Google Shopping results (SerpApi format) for the Demo provider.

They cover every accept/reject rule so the dashboard has something realistic to show before an
API key is configured. Links point at made-up item IDs.
"""

from __future__ import annotations


def _item(n, title, source, price, link, delivery="", condition="", extensions=None, snippet=""):
    item = {
        "position": n,
        "title": title,
        "source": source,
        "price": f"${price:,.2f}",
        "extracted_price": price,
        "link": link,
        "product_link": f"https://www.google.com/shopping/product/demo{n}",
        "product_id": f"demo-{n}",
        "delivery": delivery,
    }
    if condition:
        item["second_hand_condition"] = condition
    if extensions:
        item["extensions"] = extensions
    if snippet:
        item["snippet"] = snippet
    return item


RAM_RESULTS = [
    _item(1, "Samsung 64GB DDR4 3200MHz ECC Registered RDIMM M393A8G40AB2-CWE", "eBay - servergear_outlet", 189.99,
          "https://www.ebay.com/itm/100000000001", "Free delivery", "used"),
    _item(2, "SK hynix 128GB (2 x 64GB) DDR4-3200 RDIMM PC4-25600 Server Memory Kit", "Newegg.com", 519.00,
          "https://www.newegg.com/p/demo-0002", "Free delivery"),
    _item(3, "Micron 64GB DDR4-2933 LRDIMM MTA36ASF8G72LZ-2G9 Load Reduced", "Amazon.com - MemoryPro", 214.00,
          "https://www.amazon.com/dp/DEMO000003", "$6.50 delivery"),
    _item(4, "Crucial 64GB Kit (2 x 32GB) DDR4-3200 RDIMM CT2K32G4RFD432A", "B&H Photo-Video-Audio", 249.95,
          "https://www.bhphotovideo.com/c/product/demo-4", "Free delivery"),
    _item(5, "Kingston FURY Impact 64GB (2x32GB) DDR4 3200MHz SODIMM Laptop Memory", "Best Buy", 179.99,
          "https://www.bestbuy.com/site/demo-5.p", "Free delivery"),
    _item(6, "Samsung 64GB DDR4 2400MHz PC4-19200 ECC Registered Server RAM", "eBay - recycled_it", 95.00,
          "https://www.ebay.com/itm/100000000006", "Free delivery", "used"),
    _item(7, "64GB DDR4 3200 RDIMM Server RAM Brand New", "memorydealz.shop", 79.00,
          "https://memorydealz.shop/products/64gb-ddr4", "Free delivery"),
    _item(8, "Samsung 64GB DDR5 4800MHz ECC RDIMM M321R8GA0BB0-CQK", "Newegg.com - Mega Memory", 289.00,
          "https://www.newegg.com/p/demo-0008", "Free delivery"),
    _item(9, "Hynix 64GB 2Rx4 PC4-2933Y-RB2 DDR4 RDIMM - For Parts, Not Working", "eBay - parts_bin", 40.00,
          "https://www.ebay.com/itm/100000000009", "$4.99 delivery", "used"),
    _item(10, "Samsung 64GB DDR4 3200 RDIMM M393A8G40AB2-CWE", "Walmart - TechSeller", 345.00,
          "https://www.walmart.com/ip/demo/10", "Free delivery"),
    _item(11, "256GB (4x64GB) DDR4 2666 LRDIMM Server Memory Kit", "eBay - bulk_server_parts", 899.00,
          "https://www.ebay.com/itm/100000000011", "Free delivery", "used"),
    _item(12, "Samsung 64GB DDR4 ECC Registered RDIMM Server Memory", "Walmart", 150.00,
          "https://www.walmart.com/ip/demo/12", "Free delivery"),
    _item(13, "Samsung 64GB DDR4 3200 RDIMM PC4-3200AA-RB2", "eBay", 99.00,
          "https://ebay-clearance-deals.shop/item/13", "Free delivery"),
    _item(14, "Dell PowerEdge R740 2x Xeon Gold 6148 768GB (12 x 64GB) DDR4 RAM", "eBay - enterprise_liquidators",
          1200.00, "https://www.ebay.com/itm/100000000014", "Free delivery", "used"),
    _item(15, "Samsung 64GB DDR4 2666 LRDIMM PC4-2666V-LD2 M386A8K40BM2-CTD", "eBay - homelab_supply", 119.00,
          "https://www.ebay.com/itm/100000000015", "$9.00 delivery", "used"),
]  # fmt: skip

CPU_RESULTS = [
    _item(101, "AMD Ryzen Threadripper PRO 3975WX 32-Core 3.5GHz sWRX8 CPU - Not Vendor Locked",
          "eBay - workstation_pulls", 449.00, "https://www.ebay.com/itm/100000000101", "Free delivery", "used"),
    _item(102, "AMD Ryzen Threadripper PRO 3975WX 32-Core 64-Thread sWRX8 Processor", "Newegg.com - CPU Outlet",
          479.99, "https://www.newegg.com/p/demo-0102", "$9.99 delivery", "used"),
    _item(103, "AMD Threadripper PRO 3975WX 32 Core CPU Lenovo P620 Pull", "eBay - lenovo_parts", 350.00,
          "https://www.ebay.com/itm/100000000103", "Free delivery", "used"),
    _item(104, "AMD Ryzen Threadripper PRO 3975WX - Vendor Locked to Lenovo", "eBay - cheap_cpus", 280.00,
          "https://www.ebay.com/itm/100000000104", "Free delivery", "used"),
    _item(105, "AMD Threadripper PRO 3975WX CPU for parts, bent pins", "eBay - as_is_tech", 150.00,
          "https://www.ebay.com/itm/100000000105", "$5.00 delivery", "for parts"),
    _item(106, "AMD Ryzen Threadripper PRO 3975WX 32-Core, 64-Thread Unlocked Desktop Processor", "Amazon.com",
          1899.00, "https://www.amazon.com/dp/DEMO000106", "Free delivery"),
    _item(107, "Noctua NH-U14S TR4-SP3 Premium CPU Cooler for AMD Threadripper 3975WX", "Amazon.com", 89.95,
          "https://www.amazon.com/dp/DEMO000107", "Free delivery"),
    _item(108, "AMD Ryzen Threadripper PRO 5975WX 32-Core Processor", "Newegg.com", 1499.00,
          "https://www.newegg.com/p/demo-0108", "Free delivery"),
    _item(109, "Threadripper PRO 3975WX + ASUS Pro WS WRX80E-SAGE SE WIFI Motherboard Combo", "eBay - combo_deals",
          950.00, "https://www.ebay.com/itm/100000000109", "Free delivery", "used"),
    _item(110, "AMD Ryzen Threadripper PRO 3975WX 32-Core Processor", "cpu-discount-hub.com", 199.00,
          "https://cpu-discount-hub.com/products/3975wx", "Free delivery"),
    _item(111, "AMD Ryzen Threadripper PRO 3975WX Tested Working, No Bent Pins", "eBay - tested_good", 499.00,
          "https://www.ebay.com/itm/100000000111", "$8.00 delivery", "used"),
    _item(112, "AMD Threadripper PRO 3975WX 32 Core Processor - Untested", "Mercari", 300.00,
          "https://www.mercari.com/us/item/demo112", "$12.00 delivery", "used"),
]  # fmt: skip


def demo_results(query: str) -> list[dict]:
    q = query.lower()
    if "3975" in q or "threadripper" in q or "wx" in q:
        return [dict(item) for item in CPU_RESULTS]
    return [dict(item) for item in RAM_RESULTS]
