import utils

from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import re

VALID_CULTURE = ('EN', 'ZH')
VALID_CATEGORIES = ('AUTHOR', 'BEVERAGE', 'CLOTHING', 'FOOD',
                   'LOCATION', 'NAME', 'RELIGIOUS', 'SPORTS')

CULTURE = 'EN'
CATEGORY = 'LOCATION'
WRITE_XLSX = True

###################


QUERIES_ZH = {
    "author": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P106 wd:Q36180;     # "occupation of": "writer"
                    wdt:P31 wd:Q5.          # "instance of": "human" 
            FILTER (
                EXISTS { ?entity wdt:P27 wd:Q148 } ||   # "country of citizenship": "China"
                EXISTS { ?entity wdt:P27 wd:Q865 } ||   # "country of citizenship": "Republic of China"
                EXISTS { 
                    ?entity wdt:P27 ?dynasty .          # "country of citizenship": dynasty
                    ?dynasty wdt:P31 wd:Q29520.         # where dynasty has "instance of": "Chinese dynasty"  
                }         
            )
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 250
    """,
    "beverage": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity ?relation ?drinkClass.            # relation: drinkClass
            ?drinkClass wdt:P279* wd:Q40050.          # where drinkClass has "subclass of (chained)": "drink"
            FILTER (?relation IN (wdt:P31, wdt:P279)) # where relation in ("instance of", "subclass of")

            ?entity wdt:P495 ?country.                # "country of origin": country
            FILTER (?country IN (wd:Q148, wd:Q865))   # where country in ("China", "Republic of China")
            
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 500
    """,
    "clothing": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P279* wd:Q5100735.              # "subclass of (chained)": "Chinese clothing"
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 500
    """,
    "food": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P31|wdt:P279* wd:Q746549.     # ("instance of", "subclass of (chained)"): "dishes"
            ?entity wdt:P495 ?country.                # "country of origin": country
            FILTER (?country IN (wd:Q148, wd:Q865))   # where country in ("China", "Republic of China")
            
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 500
    """,
    "location": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P31 ?place.                   # "instance of": place
            FILTER (?place IN (wd:Q515, wd:Q1549591, wd:Q1208802)) 
                                                      # where place in ("city", "big city", "direct-administered municipality")
            ?entity wdt:P17 ?country.                 # "country": country
            FILTER (?country IN (wd:Q148, wd:Q865))   # where country in ("China", "Republic of China (Taiwan)")
            
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
    """,
    "name": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P31 wd:Q1093580.             # "instance of": "Chinese family name"      
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 1000
    """,
    "religious": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P31 wd:Q2680845      # "instance of": "Chinese temple"
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 1000
    """,
    "sports": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P31 ?subClub.                 # "instance of": subClub
            ?subClub wdt:P279* wd:Q847017.            # where subClubs has "subclass of (chained)": "sports club"

            ?entity wdt:P17 ?country.                 # "country": country
            FILTER (?country IN (wd:Q148, wd:Q865))   # where country in ("China", "Republic of China (Taiwan)")
            
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 1300
    """,
    "test": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P106 wd:Q36180;     # "occupation of": "writer"
                    wdt:P31 wd:Q5;          # "instance of": "human"
                    wdt:P27 ?country.       # "country of citizenship": country
            # FILTER (
            #     EXISTS { VALUES ?country { wd:Q30 wd:Q16 wd:Q408 wd:Q664 } } ||   # "country of citizenship": "China"
            #     EXISTS { 
            #         ?country wdt:P31 wd:Q6256;      # "instance of": "country"
            #                 wdt:P30 wd:Q46.         # "continent": "Europe"
            #     }         
            # )
            {
                VALUES ?country { wd:Q30 wd:Q16 wd:Q408 wd:Q664 }    # USA, Canada, Australia, New Zealand
            }
            UNION 
            {
                ?country wdt:P31 wd:Q6256;      # "instance of": "country"
                            wdt:P30 wd:Q46.     # "continent": "Europe"
            }
            SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
        }
        LIMIT 500
    """
}

QUERIES_EN = {
    "author": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P106 wd:Q36180;     # "occupation of": "writer"
                    wdt:P31 wd:Q5;          # "instance of": "human"
                    wdt:P27 ?country.       # "country of citizenship": country
            FILTER (
                EXISTS { VALUES ?country { wd:Q30 wd:Q16 wd:Q408 wd:Q664 } } ||
                EXISTS { 
                    ?country wdt:P31 wd:Q6256;      # "instance of": "country"
                            wdt:P30 wd:Q46.         # "continent": "Europe"
                }         
            )
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 500
    """,
    "beverage": """
        SELECT ?entityLabel ?entity ?views WHERE {
            ?entity ?relation ?drinkClass.            # relation: drinkClass
            ?drinkClass wdt:P279* wd:Q40050.          # where drinkClass has "subclass of (chained)": "drink"
            FILTER (?relation IN (wdt:P31, wdt:P279)) # where relation in ("instance of", "subclass of")

            ?entity wdt:P495 ?country.                # "country of origin": country
            FILTER (
                EXISTS { VALUES ?country { wd:Q30 wd:Q16 wd:Q408 wd:Q664 } } || 
                EXISTS { 
                    ?country wdt:P31 wd:Q6256;      # "instance of": "country"
                            wdt:P30 wd:Q46.         # "continent": "Europe"
                }         
            )
            
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        ORDER BY DESC(?views)
        LIMIT 500
    """,
    "clothing": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P279* wd:Q5100735.              # "subclass of (chained)": "Chinese clothing"
            SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
        }
        LIMIT 500
    """,
    "food": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P31|wdt:P279* wd:Q746549.     # ("instance of", "subclass of (chained)"): "dishes"
            ?entity wdt:P495 ?country.                # "country of origin": country
            
            FILTER (
                EXISTS { VALUES ?country { wd:Q30 wd:Q16 wd:Q408 wd:Q664 } } || 
                EXISTS { 
                    ?country wdt:P31 wd:Q6256;      # "instance of": "country"
                            wdt:P30 wd:Q46.         # "continent": "Europe"
                }         
            )
            
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 500
    """,
    "location": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P31 wd:Q1549591.                   # "instance of": place
            # FILTER (?place IN (wd:Q1549591)) 
                                                      # where place in ("city", "big city"")
            ?entity wdt:P17 ?country.                 # "country": country
            FILTER (?country IN (wd:Q30, wd:Q16, wd:Q408, wd:Q664) ||
                EXISTS { 
                    ?country wdt:P31 wd:Q6256;      # "instance of": "country"
                            wdt:P30 wd:Q46.         # "continent": "Europe"
                }  
            )            
            # FILTER (
                # EXISTS { VALUES ?country { wd:Q30 wd:Q16 wd:Q408 wd:Q664 } } 
                # EXISTS { VALUES ?country { wd:Q30 wd:Q16 wd:Q408 wd:Q664 } } ||
                # EXISTS { 
                #     ?country wdt:P31 wd:Q6256;      # "instance of": "country"
                #             wdt:P30 wd:Q46.         # "continent": "Europe"
                # }         
            # )
            
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 3000
    """,
    "name": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P31 ?subClass.          # "instance of": subClass 
            ?subClass wdt:P279* wd:Q202444.     # where subClass has "subclass of (chained)": "given name"

            ?entity wdt:P407 wd:Q1860.          # "language of work or name": "English"

            FILTER NOT EXISTS { 
                ?entity wdt:P31 wd:Q1243157.    # Exclude "double given name"
            }
            
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 500
    """,
    "religious": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P31 wd:Q2680845      # "instance of": "Chinese temple"
            SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
        }
        LIMIT 800
    """,
    "sports": """
        SELECT ?entityLabel ?entity WHERE {
            ?entity wdt:P31 ?subClub.                 # "instance of": subClub
            ?subClub wdt:P279* wd:Q847017.            # where subClubs has "subclass of (chained)": "sports club"

            ?entity wdt:P17 ?country.                 # "country": country
            FILTER (?country IN (wd:Q148, wd:Q865))   # where country in ("China", "Republic of China (Taiwan)")
            
            SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
        }
        LIMIT 500
    """
}


author_query = """
SELECT ?entityLabel ?entity WHERE {
  ?entity wdt:P106 wd:Q36180;  # "occupation of": "writer"
        wdt:P31 wd:Q5.         # "instance of": "human" 
  FILTER (
    EXISTS { ?entity wdt:P27 wd:Q148 } ||   # "country of citizenship": "China"
    EXISTS { ?entity wdt:P27 wd:Q865 } ||   # "country of citizenship": "Republic of China"
    EXISTS { 
      ?entity wdt:P27 ?dynasty .            # "country of citizenship": dynasty
      ?dynasty wdt:P31 wd:Q29520.           # where dynasty has "instance of": "Chinese dynasty"
    }         
  )
  SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
}
LIMIT 500
"""
beverage_query = """
SELECT ?entityLabel ?entity WHERE {
  ?entity ?relation ?drinkClass.            # relation: drinkClass
  ?drinkClass wdt:P279* wd:Q40050.          # where drinkClass has "subclass of (chained)": "drink"
  FILTER (?relation IN (wdt:P31, wdt:P279)) # where relation in ("instance of", "subclass of")

  ?entity wdt:P495 ?country.                # "country of origin": country
  FILTER (?country IN (wd:Q148, wd:Q865))   # where country in ("China", "Republic of China")
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
}
LIMIT 500
"""
clothing_query = """
SELECT ?entityLabel ?entity WHERE {
  ?entity wdt:P279* wd:Q5100735.              # "subclass of (chained)": "Chinese clothing"

  #?entity wdt:P495 ?country.                # "country of origin": country
  #FILTER (?country IN (wd:Q148, wd:Q865))   # where country in ("China", "Republic of China")
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
}
LIMIT 500
"""
food_query = """
SELECT ?entityLabel ?entity WHERE {
  ?entity wdt:P31|wdt:P279* wd:Q746549.     # ("instance of", "subclass of (chained)"): "dishes"

  ?entity wdt:P495 ?country.                # "country of origin": country
  FILTER (?country IN (wd:Q148, wd:Q865))   # where country in ("China", "Republic of China")
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
}
LIMIT 500
"""
location_query = """
SELECT ?entityLabel ?entity WHERE {
  ?entity wdt:P31 ?place.                   # "instance of": place
  FILTER (?place IN (wd:Q515, wd:Q1549591, wd:Q1208802)) 
                                            # where place in ("city", "big city", "direct-administered municipality")

  ?entity wdt:P17 ?country.                 # "country": country
  FILTER (?country IN (wd:Q148, wd:Q865))   # where country in ("China", "Republic of China (Taiwan)")
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
}
LIMIT 500
"""
name_query = """
SELECT ?entityLabel ?entity WHERE {
  ?entity wdt:P31 wd:Q1093580.                   # "instance of": "Chinese family name"
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
}
LIMIT 500
"""
religious_query = """
SELECT ?entityLabel ?entity WHERE {
    ?entity wdt:P31 wd:Q2680845       # "instance of": "Chinese temple"
  SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
}
LIMIT 800
"""
test_query = """SELECT ?entityLabel ?entity WHERE {
    ?entity wdt:P31|wdt:P279* wd:Q16143746       # "instance of": ""
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 800
"""
sports_query = """
SELECT ?entityLabel ?entity WHERE {
  ?entity wdt:P31 ?subClub.                   # "instance of": subClub
  ?subClub wdt:P279* wd:Q847017.            # where subClubs has "subclass of (chained)": "sports club"

  ?entity wdt:P17 ?country.                 # "country": country
  FILTER (?country IN (wd:Q148, wd:Q865))   # where country in ("China", "Republic of China (Taiwan)")
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
}
LIMIT 500
"""

def is_valid_label(label):
    # Exclude labels that start with 'Q' followed by digits (e.g., "Q12345")
    if re.match(r"^Q\d+$", label):
        return False
    # Exclude labels with non-printable/unusual characters
    if re.search(r"[^\w\s.,'’&-]", label):  
        return False
    return True

if __name__ == '__main__':
    if CULTURE not in VALID_CULTURE:
        raise ValueError(f"Invalid CULTURE: {CULTURE}. Must be one of {VALID_CULTURE}")
    if CATEGORY not in VALID_CATEGORIES:
        raise ValueError(f"Invalid CATEGORY: {CATEGORY}. Must be one of {VALID_CATEGORIES}")
    if CULTURE == 'EN':
        queries = QUERIES_EN
    else:
        queries = QUERIES_ZH
    
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    query = queries.get(CATEGORY.lower())

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query()

    results_converted = results.convert()
    
    data = [(r["entityLabel"]["value"], r["entity"]["value"]) for r in results_converted["results"]["bindings"]]
    df = pd.DataFrame(data, columns=["Entity", "Wikidata URL"])
    print(df)


    if WRITE_XLSX:
        filtered_data = list(set(row[0] for row in data if is_valid_label(row[0])))

        df_filtered = pd.DataFrame(filtered_data, columns=["Entity"])

        file_path = f'{CATEGORY.lower()}_{CULTURE.lower()}.xlsx' 
        df_filtered.to_excel(file_path, index=False)

