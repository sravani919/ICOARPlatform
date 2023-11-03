from ICOAR_core import data_collection

collector = data_collection.facebook.scraper.Collector()
results = collector.collect(["party"], 5)
print(results)
