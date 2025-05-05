from services.insights.contract_metadata_vector_handler import ContractMetadataVectorUpserter
import time
from utils.llm_status_handler.status_handler import set_llm_file_status
from utils.llm_status_handler.status_handler import set_meta_data
from services.insights.llm_call import call_llm, llm_call_for_dates, llm_call_for_jurisdiction, llm_call_for_cv, payment_due_date_validatior
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from config.config import config
from utils.hybrid_retriever.hybrid_search_retrieval import get_context_from_pinecone
from utils.logger import _log_message
import orjson
import mlflow
import psutil
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from opentelemetry import context as ot_context
mlflow.config.enable_async_logging()
# mlflow.langchain.autolog()
mlflow.openai.autolog()
    
date_pattern = r"""
\b(
    # === DOCUMENT-SPECIFIC DATE FIELD PATTERNS ===
    
    # Effective Date patterns
    (?:effective\s+date\s*[:;]?\s*(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    (?:(?:this\s+agreement\s+(?:is|shall\s+be)\s+)?effective\s+(?:as\s+of|on|from)?[\s:]*.{0,50}?(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    
    # Termination Date patterns
    (?:termination\s+date\s*[:;]?\s*(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    (?:terminates\s+(?:on|at|after|before|by)[\s:]*.{0,50}?(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    
    # Expiration Date patterns
    (?:expiration\s+date\s*[:;]?\s*(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    (?:expires\s+(?:on|at|after|before|by)[\s:]*.{0,50}?(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    
    # Term Date patterns
    (?:term\s+date\s*[:;]?\s*(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    (?:term\s+(?:shall|will)\s+(?:begin|commence|start)\s+(?:on|at|from)[\s:]*.{0,50}?(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    
    # Payment Due Date patterns
    (?:payment\s+due\s+date\s*[:;]?\s*(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    (?:payment\s+(?:is|shall\s+be)\s+due\s+(?:on|by|before|after|within)[\s:]*.{0,50}?(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    
    # Delivery Date patterns
    (?:delivery\s+date\s*[:;]?\s*(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    (?:delivery\s+(?:shall|will)\s+(?:be|occur|take\s+place)\s+(?:on|by|before|after|within)[\s:]*.{0,50}?(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    
    # Renewal Date patterns
    (?:renewal\s+date\s*[:;]?\s*(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|
    (?:(?:agreement|contract)\s+(?:shall|will|may)\s+(?:renew|be\s+renewed)\s+(?:on|as\s+of|effective)[\s:]*.{0,50}?(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4}))|

    # Section headings with dates
    (?:^|\n|\r)[\s*]*(?:effective|term|termination|expiration|renewal|payment\s+due|delivery)\s+date[\s*:-]*\s*(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:\d{1,2}(?:st|nd|rd|th)?[\s,]*)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]*\d{1,2}(?:st|nd|rd|th)?[,\s]*\d{2,4})|

    # === GENERAL DATE PATTERNS ===
    
     # ISO 8601 with time components
    \d{4}[-/]\d{1,2}[-/]\d{1,2}[T ]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})? |
    
    # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
    \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2} |
    
    # DD-MM-YYYY or DD/MM/YYYY or DD.MM.YYYY
    \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4} |
    
    # DD MM YYYY or YYYY MM DD (space separated)
    \d{1,2}\s+\d{1,2}\s+\d{4} |
    \d{4}\s+\d{1,2}\s+\d{1,2} |
    
    # Month DD, YYYY or DD Month YYYY
    (?:\d{1,2}(?:st|nd|rd|th)?[ -/]?)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ -/]?(?:\d{1,2}(?:st|nd|rd|th)?[,]?[ -/])?\d{2,4} |
    
    # Month YYYY or Month DD
    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ -/\.]\d{4} |
    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ -/\.]\d{1,2}(?:st|nd|rd|th)? |

    # Day of week with date
    (?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)[,]?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4} |
    
    # Quarter notation
    Q[1-4]\s+\d{4} |
    \d{4}\s+Q[1-4] |
    [1-4]Q\s+\d{4} |
    \d{4}\s+[1-4]Q |
    (?:First|Second|Third|Fourth|1st|2nd|3rd|4th)\s+quarter\s+\d{4} |
    \d{4}\s+(?:First|Second|Third|Fourth|1st|2nd|3rd|4th)\s+quarter |
    
    # Week notation
    Week\s+\d{1,2}\s+\d{4} |
    \d{4}\s+Week\s+\d{1,2} |
    W\d{1,2}\s+\d{4} |
    \d{4}\s+W\d{1,2} |
    W\d{1,2}-\d{4} |
    
    # Month/Year formats
    \d{1,2}/\d{4} |
    \d{1,2}-\d{4} |
    
    # Date ranges
    \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\s+(?:to|through|thru|-)\s+\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2} |
    \d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}\s+(?:to|through|thru|-)\s+\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4} |
    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\s+-\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4} |
    
    # Fiscal year notation
    FY\s*\d{2,4} |
    Fiscal\s+Year\s+\d{4} |
    Fiscal\s+\d{4} |
    
    # Chinese/Japanese year format (e.g., 令和5年10月1日)
    [令和|平成|昭和|大正|明治]\d{1,2}年\d{1,2}月\d{1,2}日 |
    
    # Timestamps with dates
    \d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)? |
    \d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)? |
    
    # Unix timestamp (10-13 digits)
    \b\d{10,13}\b |
    
    # RFC 2822 format
    (?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s+\d{2}:\d{2}(?::\d{2})?\s+(?:[+-]\d{4}|UTC|GMT|[A-Z]{3}) |
    
    # French date format (e.g., 1er janvier 2023)
    \d{1,2}(?:er|ème|e|ère)?\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4} |
    
    # Spanish date format
    \d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(?:\s+de)?\s+\d{4} |
    
    # German date format
    \d{1,2}\.\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4} |
    
    # Year-specific patterns
    
    # Year preceded by words
    (?:year|in|during|of|for|by|before|after|since|until|around|circa|ca\.?|c\.?)\s+\d{4} |
    
    # Years with era designations
    \d{1,4}\s*(?:AD|BC|BCE|CE|B\.C\.(?:E\.)?|C\.E\.|A\.D\.) |
    (?:AD|BC|BCE|CE|B\.C\.(?:E\.)?|C\.E\.|A\.D\.)\s*\d{1,4} |
    
    # Decade references
    \d{3}0s |
    \d{4}s |
    \d{4}-\d{2} |
    \d{2}s |
    (?:the\s+)?(?:twenty|nineteen|eighteen|seventeen|sixteen|fifteen|fourteen|thirteen|twelve|twenty-first|twentieth|nineteenth|eighteenth|seventeenth|sixteenth|fifteenth|fourteenth|thirteenth|twelfth)\s+(?:century|hundreds) |
    (?:the\s+)?(?:twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties) |
    
    # Year ranges
    \d{4}\s*(?:-|to|through|until|and|&|–|—)\s*\d{4} |
    \d{4}\s*(?:-|to|through|until|and|&|–|—)\s*\d{2} |
    
    # Academic/school years
    (?:academic\s+year\s+)?\d{4}[-/]\d{2,4} |
    (?:academic\s+year\s+)?\d{4}[-/]\d{2} |
    (?:AY|SY)\s*\d{4}[-/]\d{2,4} |
    
    # Centuries
    (?:(?:\d{1,2}(?:st|nd|rd|th)?|(?:twenty|nineteen|eighteen|seventeen|sixteen|fifteen|fourteen|thirteen|twelve|twenty-first|twentieth|nineteenth|eighteenth|seventeenth|sixteenth|fifteenth|fourteenth|thirteenth|twelfth))\s+century) |
    
    # Year only (must be a complete 4-digit number)
    \b\d{4}\b |
    
    # ----- NEW PATTERNS BELOW -----
    
    # Relative time expressions
    (?:(?:last|next|previous|coming|this)\s+(?:week|month|year|decade|century|spring|summer|fall|autumn|winter)) |
    (?:(?:\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|few|several|many|couple)\s+(?:second|minute|hour|day|week|fortnight|month|quarter|year|decade|century)s?\s+(?:ago|from\s+now|hence|later|earlier|before|after)) |
    
    # Specific day references
    (?:today|yesterday|tomorrow|day\s+before\s+yesterday|day\s+after\s+tomorrow) |
    
    # Days of week references
    (?:on\s+)?(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)(?:\s+(?:morning|afternoon|evening|night))? |
    
    # Month/season + year patterns
    (?:(?:early|mid|late|beginning\s+of|end\s+of)\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:of\s+)?\d{4} |
    
    # Season + year patterns
    (?:(?:early|mid|late|beginning\s+of|end\s+of)\s+)?(?:Spring|Summer|Fall|Autumn|Winter)\s+(?:of\s+)?\d{4} |
    \d{4}\s+(?:Spring|Summer|Fall|Autumn|Winter) |
    
    # Holiday references with year
    (?:Christmas|Easter|Thanksgiving|Halloween|New\s+Year(?:'s)?|Valentine's\s+Day|St\.\s+Patrick's\s+Day|Independence\s+Day|Labor\s+Day|Memorial\s+Day|Veterans\s+Day|MLK\s+Day|Martin\s+Luther\s+King\s+Day|Presidents'\s+Day|Columbus\s+Day|Hanukkah|Passover|Rosh\s+Hashanah|Yom\s+Kippur|Diwali|Eid(?:\s+al-Fitr|\s+al-Adha)?|Ramadan|Chinese\s+New\s+Year|Lunar\s+New\s+Year)\s+(?:of\s+)?\d{4} |
    \d{4}\s+(?:Christmas|Easter|Thanksgiving|Halloween|New\s+Year(?:'s)?|Valentine's\s+Day|St\.\s+Patrick's\s+Day|Independence\s+Day|Labor\s+Day|Memorial\s+Day|Veterans\s+Day|MLK\s+Day|Martin\s+Luther\s+King\s+Day|Presidents'\s+Day|Columbus\s+Day|Hanukkah|Passover|Rosh\s+Hashanah|Yom\s+Kippur|Diwali|Eid(?:\s+al-Fitr|\s+al-Adha)?|Ramadan|Chinese\s+New\s+Year|Lunar\s+New\s+Year) |
    
    # Duration expressions
    (?:for|during|over|within|in|throughout|across|spanning)\s+(?:\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|few|several|many|couple)\s+(?:second|minute|hour|day|week|month|year|decade|century)s? |
    
    # Time periods like "first half of 2023"
    (?:first|second|third|fourth|1st|2nd|3rd|4th|H1|H2|initial|early|mid|middle|late|latter|final)\s+(?:half|part|portion|quarter)\s+of\s+(?:\d{4}|the\s+year) |
    (?:Q[1-4]|[1-4]Q)\s+of\s+\d{4} |
    
    # Date and day ordinals
    (?:(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\s+(?:of\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)) |
    
    # Approximate dates
    (?:circa|ca\.|c\.|around|approximately|about|roughly|in\s+or\s+around|somewhere\s+around)\s+\d{4} |
    
    # Frequency expressions
    (?:(?:every|each)\s+(?:other\s+)?(?:second|minute|hour|day|week|month|quarter|year|decade|century|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)) |
    (?:daily|weekly|monthly|quarterly|yearly|annually|bi-weekly|bi-monthly|bi-annually|semi-annually|fortnightly) |
    
    # Time periods with prepositions
    (?:from|since|between|after|before|prior\s+to|following|as\s+of|as\s+at|starting|ending|beginning|until|till|up\s+to)\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:\d{4})? |
    
    # Age references
    (?:aged|age)\s+\d{1,3} |45 j
    \d{1,3}\s+(?:years|months|weeks|days)\s+old |
    
    # Century parts
    (?:early|mid|late|beginning\s+of|end\s+of)\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\s+century |
    (?:early|mid|late|beginning\s+of|end\s+of)\s+(?:the\s+)?(?:twentieth|nineteenth|eighteenth|seventeenth|sixteenth|fifteenth|fourteenth|thirteenth|twelfth|twenty-first)\s+century |
    
    # Time spans with "from...to" format
    from\s+\d{4}\s+to\s+\d{4} |
    from\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:\d{1,2})?\s+to\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:\d{1,2})?\s+\d{4} |
    
    # Recurring dates
    (?:every|each)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)? |
    
    # X days/weeks/months/years ago/from now
    \d{1,3}\s+(?:second|minute|hour|day|week|month|year|decade)s?\s+(?:ago|from\s+now|later|before|after|hence) |
    
    # Morning/afternoon/evening references
    (?:this|tomorrow|yesterday|next|last|previous|coming)\s+(?:morning|afternoon|evening|night) |
    
    # Weekend references
    (?:this|next|last|previous|coming)\s+weekend |
    
    # Named periods
    (?:the\s+)?(?:Great\s+Depression|Renaissance|Medieval\s+period|Middle\s+Ages|Stone\s+Age|Bronze\s+Age|Iron\s+Age|Industrial\s+Revolution|Digital\s+Age|Information\s+Age|Cold\s+War|World\s+War\s+(?:I|II|One|Two|1|2)|Victorian\s+Era|Georgian\s+Era|Edwardian\s+Era|Elizabethan\s+Era|Tudor\s+period|Roman\s+Empire|Byzantine\s+Empire|Ming\s+Dynasty|Qing\s+Dynasty|Han\s+Dynasty|Ottoman\s+Empire|Paleolithic|Mesolithic|Neolithic|post-war\s+era) |
    
    # Specific time period references
    (?:the\s+)?(?:Stone|Bronze|Iron|Dark|Middle|Golden|Modern|Post-modern|Contemporary|Colonial|Post-colonial|Post-war|Pre-war|Antebellum|Post-apocalyptic)\s+(?:Age|Era|Period) |
    
    # Business quarters and fiscal periods
    (?:Q[1-4]|[1-4]Q|first\s+quarter|second\s+quarter|third\s+quarter|fourth\s+quarter|H1|H2|first\s+half|second\s+half)\s+(?:FY)?\s*\d{2,4} |
    
    # Anniversary references
    \d{1,3}(?:st|nd|rd|th)?\s+anniversary |
    
    # Vague time references
    (?:a\s+(?:short|long)\s+(?:time|while)\s+(?:ago|from\s+now)|some\s+time\s+(?:ago|from\s+now)|in\s+recent\s+(?:days|weeks|months|years)|in\s+the\s+(?:near|distant)\s+(?:future|past))
)\b
"""
# date_pattern = r"""
# \b
# (
#     # YYYY-MM-DD or YYYY/MM/DD
#     \d{4}[-/]\d{1,2}[-/]\d{1,2} |
    
#     # DD-MM-YYYY or DD/MM/YYYY
#     \d{1,2}[-/]\d{1,2}[-/]\d{2,4} |
    
#     # Month DD, YYYY or DD Month YYYY
#     (?:\d{1,2}(?:st|nd|rd|th)?[ -/]?)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ -/]?(?:\d{1,2}(?:st|nd|rd|th)?[,]?[ -/])?\d{2,4} |
    
#     # Month YYYY or Month DD
#     (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ -/]\d{4} |
#     (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ -/]\d{1,2} |

#     # Year only
#     \b\d{4}\b
# )
# \b
# """
jurisdiction_regex = r"""
Comprehensive regex pattern for identifying jurisdiction and governing law clauses in legal documents.

This pattern captures various forms of jurisdiction references including:
- Governing law clauses
- Jurisdiction and venue clauses
- Conflict of laws provisions
- Forum selection clauses
- Arbitration seat designations
- Court references
- Legal proceeding locations
- Applicable law statements

Usage:
    import re
    jurisdiction_pattern = re.compile(jurisdiction_regex, re.VERBOSE | re.IGNORECASE)
    matches = jurisdiction_pattern.findall(document_text)
"""

jurisdiction_regex = r"""
Comprehensive regex pattern for identifying jurisdiction and governing law clauses in legal documents.

This pattern captures:
- Governing law clauses
- Jurisdiction and venue clauses
- Conflict of laws provisions
- Forum selection clauses
- Arbitration seat designations
- Specific countries, states, provinces, and territories worldwide
- Court references
- Legal proceeding locations
- Applicable law statements

Last updated: 2025-05-05
Author: sanket2992 (with assistance)
"""

jurisdiction_regex = r"""(?xi)\b(
    # Governing law clauses
    governed\s+by\s+the\s+laws?\s+of |
    governing\s+law(?:\s+shall\s+be)?(?:\s+of)? |
    shall\s+be\s+governed\s+by(?:\s+the\s+laws?\s+of)? |
    construed\s+(?:and\s+enforced\s+)?(?:in\s+accordance\s+)?with\s+the\s+laws?\s+of |
    interpreted\s+(?:in\s+accordance\s+)?(?:with|under)\s+the\s+laws?\s+of |
    enforced\s+(?:in\s+accordance\s+)?with\s+the\s+laws?\s+of |
    applicable\s+laws?\s+(?:shall\s+be\s+that|are\s+those)\s+of |
    laws?\s+applicable\s+(?:to|in) |

    # Subject‐to/jurisdiction clauses
    subject\s+to\s+(?:the\s+)?jurisdiction\s+of |
    (?:exclusive|non[-\s]?exclusive|sole)\s+jurisdiction\s+of |
    (?:proper|applicable)\s+venue(?:\s+shall\s+be\s+in)? |
    venue\s+shall\s+be\s+in |
    jurisdiction\s+and\s+venue\s+(?:shall|will)\s+(?:be|lie|exist|vest)\s+(?:exclusively\s+)?in |
    courts?\s+of\s+competent\s+jurisdiction\s+(?:in|of|located\s+in) |
    competent\s+jurisdiction\s+(?:in|of|located\s+in) |

    # Legal/forum clauses
    (?:legal|judicial)\s+forum(?:\s+of)? |
    forum\s+for\s+(?:any|all)\s+disputes? |
    forum\s+(?:selection|clause) |
    disputes?\s+(?:shall|will|may)\s+be\s+(?:resolved|determined|heard|litigated|adjudicated|decided|settled)\s+
        (?:in\s+accordance\s+with\s+)?(?:the\s+)?laws?\s+of |
    choice\s+of\s+law |
    conflict\s+of\s+laws? |
    without\s+(?:regard\s+to|giving\s+effect\s+to)\s+(?:its\s+)?conflict\s+of\s+laws?\s+(?:principles|provisions|rules) |
    irrespective\s+of\s+conflict\s+of\s+laws |
    excluding\s+(?:the\s+application\s+of\s+)?(?:any\s+)?conflict\s+of\s+laws?\s+(?:principles|provisions|rules) |

    # Submission clauses
    (?:submitted|subject)\s+to\s+(?:the\s+)?(?:courts?|jurisdiction|tribunals?)(?:\s+of)? |
    submits?\s+(?:themselves|itself|himself|herself)\s+to\s+the\s+jurisdiction\s+of |
    consents?\s+to\s+the\s+jurisdiction\s+of |
    
    # Legal proceedings clauses
    (?:any|all)\s+(?:legal\s+)?proceedings?\s+(?:shall|must|will|may)\s+be\s+brought\s+(?:exclusively\s+)?in |
    (?:any|all)\s+(?:actions?|suits?|claims?|proceedings?)\s+(?:arising|resulting)\s+
        (?:out\s+of|from|under|in\s+connection\s+with)\s+this |
    (?:any|all)\s+disputes?\s+(?:arising|resulting)\s+
        (?:out\s+of|from|under|in\s+connection\s+with)\s+this |
    (?:any|all)\s+(?:legal\s+)?(?:actions?|proceedings?)\s+to\s+enforce |
    
    # Seat/place of arbitration/jurisdiction
    place\s+of\s+(?:arbitration|jurisdiction|performance) |
    seat\s+of\s+(?:arbitration|jurisdiction) |
    arbitration\s+(?:shall|will)\s+be\s+conducted\s+in |
    arbitration\s+proceedings\s+shall\s+take\s+place\s+in |
    arbitration\s+venue\s+shall\s+be |
    
    # Court types and specific references
    (?:federal|state|district|circuit|supreme|high|appellate|superior|county|provincial|municipal)\s+courts?\s+
        (?:sitting|located)\s+in |
    courts?\s+of\s+the\s+(?:state|province|district|county|city|country|nation)\s+of |
    
    # Country-specific jurisdictional terms
    (?:chancery|common\s+law|crown|administrative|commercial|civil|criminal|family|probate|labor|tax|bankruptcy)\s+
        courts?\s+of |
    
    # "Courts of X" or "laws of X" with multi‐word or acronym regions
    (?:courts?|laws?|tribunals?)\s+of\s+
        (?:[A-Z][a-zA-Z\-'.]*(?:\s+(?:and|&|or)\s+[A-Z][a-zA-Z\-'.]*)*
            (?:\s+(?:County|State|Province|Country|Nation|Republic|Kingdom|Government|Commonwealth|Federation|Union|Emirate|Territory))?|
        (?:USA|US|U\.S\.|U\.S\.A\.|UK|U\.K\.|UAE|U\.A\.E\.|EU|E\.U\.|PRC|P\.R\.C\.|ROK|R\.O\.K\.|APAC|EMEA|LATAM))
    |
    
    # Specific country and region references
    # Major countries and regions
    (?:in|of|under\s+the\s+laws?\s+of|subject\s+to\s+the\s+jurisdiction\s+of)\s+
    (?:
        # Countries (major ones and common legal jurisdictions)
        Afghanistan|Albania|Algeria|Andorra|Angola|Antigua\s+and\s+Barbuda|Argentina|Armenia|Australia|Austria|
        Azerbaijan|Bahamas|Bahrain|Bangladesh|Barbados|Belarus|Belgium|Belize|Benin|Bhutan|Bolivia|
        Bosnia\s+and\s+Herzegovina|Botswana|Brazil|Brunei|Bulgaria|Burkina\s+Faso|Burundi|Cabo\s+Verde|Cambodia|
        Cameroon|Canada|Central\s+African\s+Republic|Chad|Chile|China|Colombia|Comoros|Congo|Costa\s+Rica|
        Croatia|Cuba|Cyprus|Czech\s+Republic|Denmark|Djibouti|Dominica|Dominican\s+Republic|
        East\s+Timor|Ecuador|Egypt|El\s+Salvador|Equatorial\s+Guinea|Eritrea|Estonia|Eswatini|Ethiopia|
        Fiji|Finland|France|Gabon|Gambia|Georgia|Germany|Ghana|Greece|Grenada|Guatemala|Guinea|Guinea-Bissau|
        Guyana|Haiti|Honduras|Hungary|Iceland|India|Indonesia|Iran|Iraq|Ireland|Israel|Italy|
        Jamaica|Japan|Jordan|Kazakhstan|Kenya|Kiribati|Korea|Kosovo|Kuwait|Kyrgyzstan|Laos|Latvia|
        Lebanon|Lesotho|Liberia|Libya|Liechtenstein|Lithuania|Luxembourg|Madagascar|Malawi|Malaysia|
        Maldives|Mali|Malta|Marshall\s+Islands|Mauritania|Mauritius|Mexico|Micronesia|Moldova|Monaco|
        Mongolia|Montenegro|Morocco|Mozambique|Myanmar|Namibia|Nauru|Nepal|Netherlands|New\s+Zealand|
        Nicaragua|Niger|Nigeria|North\s+Korea|North\s+Macedonia|Norway|Oman|Pakistan|Palau|Palestine|
        Panama|Papua\s+New\s+Guinea|Paraguay|Peru|Philippines|Poland|Portugal|Qatar|Romania|Russia|
        Rwanda|Saint\s+Kitts\s+and\s+Nevis|Saint\s+Lucia|Saint\s+Vincent\s+and\s+the\s+Grenadines|Samoa|
        San\s+Marino|Sao\s+Tome\s+and\s+Principe|Saudi\s+Arabia|Senegal|Serbia|Seychelles|Sierra\s+Leone|
        Singapore|Slovakia|Slovenia|Solomon\s+Islands|Somalia|South\s+Africa|South\s+Korea|South\s+Sudan|
        Spain|Sri\s+Lanka|Sudan|Suriname|Sweden|Switzerland|Syria|Taiwan|Tajikistan|Tanzania|Thailand|
        Togo|Tonga|Trinidad\s+and\s+Tobago|Tunisia|Turkey|Turkmenistan|Tuvalu|Uganda|Ukraine|
        United\s+Arab\s+Emirates|United\s+Kingdom|United\s+States(?:\s+of\s+America)?|Uruguay|Uzbekistan|
        Vanuatu|Vatican\s+City|Venezuela|Vietnam|Yemen|Zambia|Zimbabwe|
        Hong\s+Kong|Macau|Puerto\s+Rico|Scotland|Northern\s+Ireland|Wales|England|
        
        # US States
        Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|
        Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|
        Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|
        New\s+Hampshire|New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|
        Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|South\s+Dakota|
        Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming|
        District\s+of\s+Columbia|D\.C\.|Washington\s+D\.C\.|
        
        # Canadian Provinces/Territories
        Alberta|British\s+Columbia|Manitoba|New\s+Brunswick|Newfoundland\s+and\s+Labrador|
        Northwest\s+Territories|Nova\s+Scotia|Nunavut|Ontario|Prince\s+Edward\s+Island|Quebec|
        Saskatchewan|Yukon|
        
        # Australian States/Territories
        New\s+South\s+Wales|Queensland|South\s+Australia|Tasmania|Victoria|Western\s+Australia|
        Australian\s+Capital\s+Territory|Northern\s+Territory|
        
        # Indian States/Territories
        Andhra\s+Pradesh|Arunachal\s+Pradesh|Assam|Bihar|Chhattisgarh|Goa|Gujarat|Haryana|
        Himachal\s+Pradesh|Jharkhand|Karnataka|Kerala|Madhya\s+Pradesh|Maharashtra|Manipur|
        Meghalaya|Mizoram|Nagaland|Odisha|Punjab|Rajasthan|Sikkim|Tamil\s+Nadu|Telangana|
        Tripura|Uttar\s+Pradesh|Uttarakhand|West\s+Bengal|Andaman\s+and\s+Nicobar\s+Islands|
        Chandigarh|Dadra\s+and\s+Nagar\s+Haveli\s+and\s+Daman\s+and\s+Diu|Delhi|Jammu\s+and\s+Kashmir|
        Ladakh|Lakshadweep|Puducherry|
        
        # UK Countries/Regions
        England|Scotland|Wales|Northern\s+Ireland|
        
        # Chinese Provinces/Regions
        Anhui|Beijing|Chongqing|Fujian|Gansu|Guangdong|Guangxi|Guizhou|Hainan|Hebei|
        Heilongjiang|Henan|Hubei|Hunan|Inner\s+Mongolia|Jiangsu|Jiangxi|Jilin|Liaoning|
        Ningxia|Qinghai|Shaanxi|Shandong|Shanghai|Shanxi|Sichuan|Tianjin|Tibet|Xinjiang|
        Yunnan|Zhejiang|
        
        # Brazilian States
        Acre|Alagoas|Amapá|Amazonas|Bahia|Ceará|Espírito\s+Santo|Goiás|Maranhão|
        Mato\s+Grosso|Mato\s+Grosso\s+do\s+Sul|Minas\s+Gerais|Pará|Paraíba|Paraná|
        Pernambuco|Piauí|Rio\s+de\s+Janeiro|Rio\s+Grande\s+do\s+Norte|Rio\s+Grande\s+do\s+Sul|
        Rondônia|Roraima|Santa\s+Catarina|São\s+Paulo|Sergipe|Tocantins|
        
        # Mexican States
        Aguascalientes|Baja\s+California|Baja\s+California\s+Sur|Campeche|Chiapas|
        Chihuahua|Coahuila|Colima|Durango|Guanajuato|Guerrero|Hidalgo|Jalisco|
        México|Mexico\s+City|Michoacán|Morelos|Nayarit|Nuevo\s+León|Oaxaca|Puebla|
        Querétaro|Quintana\s+Roo|San\s+Luis\s+Potosí|Sinaloa|Sonora|Tabasco|
        Tamaulipas|Tlaxcala|Veracruz|Yucatán|Zacatecas|
        
        # German States
        Baden-Württemberg|Bavaria|Berlin|Brandenburg|Bremen|Hamburg|Hesse|
        Lower\s+Saxony|Mecklenburg-Vorpommern|North\s+Rhine-Westphalia|Rhineland-Palatinate|
        Saarland|Saxony|Saxony-Anhalt|Schleswig-Holstein|Thuringia|
        
        # Russian Regions (Federal Subjects)
        Moscow|Saint\s+Petersburg|Adygea|Altai|Bashkortostan|Buryatia|Chechnya|
        Chuvashia|Dagestan|Ingushetia|Kabardino-Balkaria|Kalmykia|Karachay-Cherkessia|
        Karelia|Khakassia|Komi|Mari\s+El|Mordovia|North\s+Ossetia-Alania|Tatarstan|
        Tuva|Udmurtia|Sakha|Yakutia
    )
)\b"""

contract_value_regex = r"""
(  # Removed the enclosing \b(...)\b
    # Symbol-Based Currency Amounts
    (?:[\$\€\£\¥\₹]|AED|USD|INR|GBP|EUR|CNY|RMB|JPY|SAR)\s?
    (?:
        \d{1,3}(?:,\d{3})*(?:\.\d+)? |
        \d+(?:\.\d+)?\s*(?:million|billion|crore|lakh|thousand)?
    )
|
    # Name-Based Currency Amounts
    (?:
        (?:(?:[Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine|[Tt]en|
        [Ee]leven|[Tt]welve|[Tt]hirteen|[Ff]ourteen|[Ff]ifteen|[Ss]ixteen|[Ss]eventeen|
        [Ee]ighteen|[Nn]ineteen|[Tt]wenty|[Tt]hirty|[Ff]orty|[Ff]ifty|[Ss]ixty|[Ss]eventy|
        [Ee]ighty|[Nn]inety|[Oo]ne\s+hundred|[Oo]ne\s+thousand|
        [Oo]ne\s+million|[Oo]ne\s+billion|[Oo]ne\s+crore|[Oo]ne\s+lakh)(\s+and\s+)?)*
        (?:[Mm]illion|[Bb]illion|[Tt]housand|[Cc]rore|[Ll]akh)?
    )
    \s+
    (?:dollars?|pounds?|euros?|rupees?|dirhams?|yuan|yen)
)
"""

metadata_hybrid_queries = [
    {"Effective Date": {
    "What is the effective date of the contract?": """Instructions:
        ### Find the date the agreement starts or becomes active. Look for phrases like "effective date," "start date," or "commencement date."
        ### Return the date in "YYYY-MM-DD" format. 
        ### If no date is found or it's unclear, return "null" as a string.
        ### Expected Output:
        {"Effective Date":""}"""}
        },
{"Termination Date": {
    "What is the termination date of the contract?": """Instructions:
        ### Find the date the agreement ends or is terminated. Look for phrases like "termination date," "end date," or "expiration of term."
        ### Return the date in "YYYY-MM-DD" format.
        ### If no date is found or it's unclear, return "null" as a string.
        ### Expected Output:
        {"Termination Date":""}"""}},
{"Renewal Date": {
    "What is the renewal date of the contract?": """Instructions:
        ### Find the date the agreement renews or extends. Look for phrases like "renewal date" or "extension date."
        ### Return the date in "YYYY-MM-DD" format.
        ### If no date is found or it's unclear, return "null" as a string.
        ### Expected Output:
        {"Renewal Date":""}"""}},
{"Expiration Date": {
    "What is the expiration date of the contract?": """Instructions:
         ### Find the date the agreement expires. Look for phrases like "expiration date," "contract ends on," or "end of term."
         ### Return the date in "YYYY-MM-DD" format.
         ### If no date is found or it's unclear, return "null" as a string.
         ### Expected Output:
         {"Expiration Date":""}"""}},
{"Delivery Date": {
    "What is the delivery date of the contract?": """Instructions:
        ### Find the date when delivery is scheduled or completed. Look for phrases like "delivery date," "completion date," or "scheduled delivery."
        ### Return the date in "YYYY-MM-DD" format.
        ### If no date is found or it's unclear, return "null" as a string.
        ### Expected Output:
        {"Delivery Date":""}"""}},
{"Term Date": {
        "What is the term date of the agreement?": """Instructions:
            ### Look for the "start date" of the agreement and return it in "YYYY-MM-DD" format if found.
            ### If the "start date" is not available or unclear, return "null" as a string.
            ### Expected Output:
            {"Term Date":""}"""
    }},
{ "Jurisdiction": {
    "What is the jurisdiction?" : """Instructions:
        ###Description: "The legal authority or location under which the contract is governed. This specifies the applicable laws and courts that will handle any disputes related to the agreement.",
        ###Extract the jurisdiction from the text.
        If the contract contains a “governed by” clause, use that.
        Otherwise, if it refers to being valid, permissible, or enforceable “under <Place> law”, treat <Place> as the jurisdiction.
        Return only the jurisdiction name, or "null" if there’s no location at all.
        ###Expected Output:
        {"Jurisdiction":""}"""}},

{ "Contract Value": {
    "What is the contract value?" : """Instructions:
        ###Description": "The financial worth of the contract, typically expressed as a monetary amount. This might include the total payment, installment amounts, or overall budget agreed upon."
        ###Extract the contract value from the text, returning only the value as integer or "null" as string if it is not mentioned, cannot be inferred, or is too vague, with no extra text or formatting.
        ###Strictly don't include any currency symbols or words.
        ###Expected Output:
        {"Contract Value":""}"""}}
]


date_extraction_instructions = """
Extract the following date fields from a contract or agreement document with enhanced accuracy.
All extracted dates must be returned strictly in the format: "YYYY-MM-DD" (4-digit year, 2-digit month, 2-digit day).
If a date is not present or cannot be confidently extracted, return the string "null".

1. Effective Date:
   - Identify the date when the agreement begins, becomes active, or is deemed to take effect.
   - Common keywords/phrases: "effective date", "this agreement is made effective as of", "commencement date", 
     "start date", "as of the date", "executed on", "entered into on", "dated as of".
   - This date often appears in the opening paragraphs of an agreement.

2. Termination Date:
   - Identify the date when the agreement may be terminated before its natural expiration.
   - Look for keywords: "termination date", "this agreement shall terminate on", "early termination", 
     "right to terminate on", "may be terminated on", "terminate prior to expiration".
   - IMPORTANT: Distinguish from expiration date. Termination typically refers to an early or conditional ending.

3. Renewal Date:
   - Identify the date when the agreement automatically renews or is extended.
   - Keywords: "renewal date", "renewed on", "extension period begins", "contract shall automatically renew on",
     "renewal term commences", "option to renew on", "renewal period starting".
   - Check for phrases indicating automatic or optional renewal periods.

4. Expiration Date:
   - Identify the final date after which the agreement naturally ends at the conclusion of its term.
   - Keywords: "expiration date", "this agreement expires on", "contract ends on", "end of term",
     "shall expire on", "valid until", "in effect through", "shall continue until".
   - IMPORTANT: Distinct from termination date. Expiration refers to the natural end of the contract term.
   - Look for language indicating the end of the "Term" section.

5. Delivery Date:
   - Identify the date when goods/services are scheduled to be or were actually delivered.
   - Keywords: "delivery date", "delivery shall be made on", "completion date", "scheduled delivery", 
     "delivered on", "due date", "shipment date", "to be provided by".
   - May appear in sections related to services, deliverables, or performance schedules.

6. Term Date:
   - Identify the date the contractual term begins (often similar to effective date).
   - Keywords: "term begins", "term of this agreement shall commence on", "start date of the term",
     "term commencement", "initial term begins", "contract term starts".
   - Check the dedicated "Term" section of the agreement.

CRITICAL - Handle Reference-Based Dates:
- For dates described relatively (e.g., "15 years after the effective date", "30 days following delivery"):
  1. Identify the reference date (e.g., effective date, delivery date)
  2. Identify the time period (e.g., 15 years, 30 days)
  3. Calculate the actual date if possible, otherwise return "calculated:{reference}+{period}"
  4. Example: If effective date is "2020-01-01" and expiration is "15 years after effective date", 
     return "2035-01-01" for expiration date

Context Awareness:
- Pay attention to document structure - dates often appear in specific sections
- Check for amended dates that may supersede earlier mentions
- For multiple possible dates, prioritize the most specific and contextually appropriate one
- Differentiate between "may terminate after X date" (termination right) versus actual termination date

Output Format (as JSON string):
{
  "Effective Date": "",
  "Termination Date": "",
  "Renewal Date": "",
  "Expiration Date": "",
  "Delivery Date": "",
  "Term Date": ""
}

Formatting Rules:
- All dates must follow the "YYYY-MM-DD" format (e.g., "2025-04-11")
- If a date is not clearly mentioned or cannot be confidently determined, return "null" as a string
- For calculated dates, if specific calculation cannot be performed but reference is clear, 
  return in format "calculated:{reference}+{period}" (e.g., "calculated:effective_date+15_years")
"""

jurisdiction_instruction ="""  
**Task**: Extract **jurisdiction(s)** from the contract text.  
- A contract may specify **multiple jurisdictions** for different purposes (e.g., "governed by California laws" and "disputes resolved in Delaware courts").  
- Return **all valid jurisdictions** as a **single string**, joined by "and" (e.g., `"California and Delaware"`).  
- Include jurisdictions **only if explicitly stated** (e.g., "laws of [Location]", "courts of [Location]").  
- Return "null" if:  
  - No jurisdiction is mentioned.  
  - Jurisdiction is vague (e.g., "applicable laws", "competent courts" without specifics).  
  - Ambiguous or inferred jurisdictions.  

**Output Format**:  
Return a JSON object with the key "Jurisdiction". 
-**Strictly do not include "```" or "json" or any markers in your response.**
Examples:  
{"Jurisdiction": "Texas"}  # Single jurisdiction  
{"Jurisdiction": "Singapore and UK"}  # Multiple jurisdictions  
{"Jurisdiction": "null"}  # No valid jurisdiction"""

contract_value_instructions = """
**Task**: Calculate and extract the **total contract value** from the text.  
- Contract value refers to the financial worth of the agreement, including totals, installments, or budgets.  
- **Return only the numerical value** as an integer (remove commas/currency symbols).  
- **Handle complex cases**:  
  - If multiple amounts are stated (e.g., "$500,000 upfront and $300,000 annually"), **sum them** and return the total.  
  - If a **total value** is explicitly mentioned, use it instead of partial amounts.  
- Return "null" if:  
  - No value is mentioned.  
  - Values are ambiguous (e.g., "subject to funding", "confidential").  
  - Non-numeric descriptions (e.g., "to be negotiated").  

**Output Format**:
-**Strictly do not include "```" or "json" or any markers in your response.**  
{"Contract Value": "<integer_total_or_null>"}  
"""

METADATA_EXTRACTION_PROMPTS = [

    {
    "What is this contract is about, the scope of work, the purpose of this contract": """Instructions:
        ###Description": "A 1-2 line summary of the context provided "
        ###The scope of work in a contract defines the specific tasks, deliverables, and timelines associated with the project or service being performed, with no extra text or formatting.
        ###Expected Output:
        {"Scope of Work":""}"""
    },

    {"What is the title of the contract?":"""Instructions:
        ###Description": "The official title or name of the contract, often found at the top of the document or in the introductory clauses. This is typically a concise label summarizing the purpose of the agreement (e.g., 'Service Agreement' or 'Purchase Contract')."
        ###Extract the title of the contract from the text, returning only the title or "null" as string if it is not mentioned, cannot be inferred, or is too vague, with no extra text or formatting.
        ###Expected Output:
        {"Title of the Contract":""}"""},
    {"What is risk mitigation score?" : """Instructions:
        ###Description": "A numerical or descriptive evaluation of potential risks and their mitigation strategies within the contract."
        ###Extract the risk mitigation score from the text, returning only the score or "null" as string if it is not mentioned, cannot be inferred, or is too vague, with no extra text or formatting.
        ###Expected Output:
        {"Risk Mitigation Score":""}"""},
    {"what are the parties involved?" : """
        Instructions:
            ### Description: Extract the actual names or unique identifiers of the parties bound by the contract, as explicitly mentioned in the provided text (e.g., "John Smith", "ABC Corp."). These may be full names, company names, or other explicit identifiers.
            ### If NO actual names or identifiers are present, but aliases/roles (such as 'Seller', 'Buyer', 'Purchaser', etc.) are mentioned, return the aliases as they appear, comma-separated.
            ### If NEITHER actual names/identifiers nor aliases/roles are present, return "null".
            ### Return only the parties or "null" as a string in the following JSON format, without extra text or formatting.
            ### Examples:
            #   Input: "This contract is made between John A. Smith (the Seller) and XYZ Corp (the Purchaser)."
            #   Output: {"Parties Involved":"John A. Smith, XYZ Corp"}
            #   Input: "This agreement is between the Seller and the Purchaser."
            #   Output: {"Parties Involved":"Seller, Purchaser"}
            #   Input: "This agreement is made this day."
            #   Output: {"Parties Involved":"null"}
            ### Expected Output:
            {"Parties Involved":""}
        """},
    {"what is the contract Type?" : """Instructions:
        Accurately classify contract-related inputs based on their context and purpose. Match them to one of the predefined contract types when applicable, but also accommodate new, clearly defined categories beyond the list when necessary.

        ### Guidelines for Classification:

            - Contextual Analysis: Analyze the input thoroughly to understand the intent, content, and purpose of the contract.

            - Flexible Classification: Start by comparing the input against the predefined contract types, but if no match is apparent, identify and return a new, logically appropriate classification.

            - Consistency Across Runs: Implement deterministic methods so that identical inputs always yield consistent results.

            - No Forced Matching: If the input does not fit any logical category, predefined or newly identified, return "null"

        ###Key Steps:

            - Step 1: Understand the Context: Carefully read and grasp the details and intent of the input.

            - Step 2: Compare to Predefined Types: Match the input to the provided contract type list if applicable.

            - Step 3: Extend When Necessary: If no predefined type fits, propose a new logical classification.

            - Step 4: Ensure Consistency: Use a clear, logical framework to make classifications repeatable and reliable.

            - Step 5: Output Determination: Output either the contract type (predefined or newly identified) or "null".


        ### Contract Type List:
        1. Business Contracts:
        - Partnership
        - Employment
        - Non-Disclosure Agrrement (NDA)
        - Sales
        - Service
        - Marketing
        - Supply
        - Franchise

        2. Real Estate Contracts:
        - Lease
        - Purchase
        - Mortgage
        - Rental

        3. Financial Contracts:
        - Loan
        - Credit
        - Investment

        4. Intellectual Property Contracts:
        - Licensing
        - Assignment

        5. Construction Contracts:
        - Construction
        - Subcontractor

        6. Technology and IT Contracts:
        - Software License
        - Service Level Agrrement (SLA)

        7. Government Contracts:
        - Procurement
        - Grants and Funding

        8. Personal Contracts:
        - Prenuptial
        - Separation
        - Settlement

        9. Sales and Purchase Contracts:
        - Bill of Sale
        - Purchase Orders

        10. Employment and Labor Contracts:
        - Collective Bargaining
        - Independent Contractor

        11. Healthcare Contracts:
        - Physician Employment
        - Hospital Service

        12. Insurance Contracts:
        - Policy
        - Reinsurance


        # Output Format

        - If a contract type is identified: 
        ```json
        {"Contract Type": "<exact_or_new_match>"}
        ```

        - If no match is found: 
        ```json
        {"Contract Type": ""}
        ```

        # Examples

        - **Example 1:**
        - Input: "This agreement outlines the provision of digital marketing services to promote the company's products."
        - Output: 
            ```json
            {"Contract Type": "Marketing"}
            ```

        - **Example 2:**
        - Input: "This contract grants the use of software for a specified duration and is subject to licensing terms."
        - Output:
            ```json
            {"Contract Type": "Software License"}
            ```

        - **Example 3:**
        - Input: "An agreement established between two parties regarding personal property."
        - Output:
            ```json
            {"Contract Type": ""}
            ```
            
        # Notes
        - Ensure consistency and accuracy by applying clear, logical rules to classification.

        - Consider context, purpose, and intent when determining classifications beyond the predefined list.

        - Case-insensitive matching is allowed, but the exact terminology should always be reflected in the output.

        - Do not guess or force classifications—return "null" when input is ambiguous or lacks sufficient detail.

        ### Expected Output:
        {"Contract Type":"<exact_match_from_list>"} or {"Contract Type":""}"""},
    {"what is the contract duration?" : """Instructions:
        ###Description": "The total time span for which the contract is valid. This may include start and end dates or a specific date"
        ###Extract the contract duration from the text, returning only the duration or "null" as string if it is not mentioned, cannot be inferred, or is too vague, with no extra text or formatting.
        ###Expected Output:
        {"Contract Duration":""}"""},
    {
        "What is the version of this agreement?": """Instructions:
            ###Description": "The version ID, contract number, or revision number of the agreement, used to track updates, amendments, or changes. This helps identify the specific iteration of the document."
            ###Extract the version ID, contract number, or any related identifier from the text, returning only the version or contract number or "null" as a string if it is not mentioned, cannot be inferred, or is too vague, with no extra text or formatting.
            ###Expected Output:
            {"Version Control":""}"""
            
    }
]



DOCUMENT_SUMMARY_AND_CHUNK_INDEX = config.DOCUMENT_SUMMARY_AND_CHUNK_INDEX
PINECONE_API_KEY = config.PINECONE_API_KEY
MODULE_NAME = "meta_data_extractor.py"



class MetaDataExtractor:

    def __init__(self, logger=None):
        self.logger = logger
        self.metadata_vector_handler = ContractMetadataVectorUpserter(logger)
        self.in_queue = False



    def _log_message(self, message: str, function_name: str) -> str:
        """
        Constructs a structured log message.
        """
        return _log_message(message, function_name, MODULE_NAME)
    
    def clean_and_split_sentences(self, text):
        """
        Cleans markdown/special characters and splits text into sentences.
        """
        # Remove markdown characters and special characters
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Remove markdown images
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)   # Remove markdown links
        text = re.sub(r'[*_~`#+\-=|>\[\](){}!\\/<>]', '', text)  # Strip other markdown symbols
        text = re.sub(r'\s+', ' ', text).strip()  # Collapse multiple spaces

        # Split on . followed by optional markdown or spacing chars
        sentences = re.split(r'\.(?:\s|\||\*|,)*', text)
        
        # Remove any empty strings from result
        return [s.strip() for s in sentences if s.strip()]
    
    
    def extract_regex_chunks_with_words(self, chunks, regex_pattern):
        """
        Extract 1 sentence before, the sentence with the RE match, and 1 sentence after.
        Ensure the max length of sentences before and after is 25 words.
        Avoid overlap and repetitions of words and sentences.
        """
        combined_text = []
        seen_sentences = set()  # Track sentences to avoid repetition
        chunks = "\n".join([chunk for chunk in chunks])
        sentences = self.clean_and_split_sentences(chunks)
        self.logger.info(self._log_message(f"Length of sentences:{len(sentences)} ", "extract_regex_chunks_with_words"))

        i = 0
        if regex_pattern == "date_pattern":
            regex_pattern = date_pattern
            while i < len(sentences):
                if re.search(regex_pattern, sentences[i], re.VERBOSE):
                    print(f"Matched Sentence {i}: {sentences[i]}\n\n")
                    # Extract 1 sentence before, the matched sentence, and 1 sentence after
                    start = max(0, i - 2)
                    end = min(len(sentences), i + 3)  # Include the matched sentence and 1 after
                    selected_sentences = sentences[start:end]

                    # Limit the length of sentences before and after to 25 words
                    trimmed_sentences = []
                    for j, sent in enumerate(selected_sentences):
                        if sent not in seen_sentences:  # Avoid duplicate sentences
                            words = word_tokenize(sent)
                            if j == 0 and i > 0:  # Sentence before
                                trimmed_sentences.append(" ".join(words[-45:]))
                            elif j == len(selected_sentences) - 1 and i + 1 < len(sentences):  # Sentence after
                                trimmed_sentences.append(" ".join(words[:45]))
                            else:  # Matched sentence
                                trimmed_sentences.append(sent)
                            seen_sentences.add(sent)  # Mark sentence as seen

                    combined_text.extend(trimmed_sentences)
                    i = end  # Skip to the sentence after the extracted range to avoid overlap
                else:
                    i += 1

            # Combine all matched text into a single chunk
            if combined_text:
                filtered = [{"chunk_number": "combined", "text": " ".join(combined_text)}]
                self.logger.info(self._log_message(f"Filtered on line 498: {filtered}", "extract_regex_chunks_with_words"))
                return filtered[0]['text']
            return []
        if regex_pattern == "jurisdiction_regex":
            regex_pattern = jurisdiction_regex
            while i < len(sentences):
                if re.search(regex_pattern, sentences[i], re.VERBOSE):
                    print(f"Matched Sentence {i}: {sentences[i]}\n\n")
                    # Extract 1 sentence before, the matched sentence, and 1 sentence after
                    start = max(0, i - 1)
                    end = min(len(sentences), i + 2)  # Include the matched sentence and 1 after
                    selected_sentences = sentences[start:end]

                    # Limit the length of sentences before and after to 25 words
                    trimmed_sentences = []
                    for j, sent in enumerate(selected_sentences):
                        if sent not in seen_sentences:  # Avoid duplicate sentences
                            words = word_tokenize(sent)
                            if j == 0 and i > 0:  # Sentence before
                                trimmed_sentences.append(" ".join(words[-30:]))
                            elif j == len(selected_sentences) - 1 and i + 1 < len(sentences):  # Sentence after
                                trimmed_sentences.append(" ".join(words[:30]))
                            else:  # Matched sentence
                                trimmed_sentences.append(sent)
                            seen_sentences.add(sent)  # Mark sentence as seen

                    combined_text.extend(trimmed_sentences)
                    i = end  # Skip to the sentence after the extracted range to avoid overlap
                else:
                    i += 1

            # Combine all matched text into a single chunk
            if combined_text:
                filtered = [{"chunk_number": "combined", "text": " ".join(combined_text)}]
                self.logger.info(self._log_message(f"Filtered on line 530: {filtered}", "extract_regex_chunks_with_words"))
                return filtered[0]['text']
            return []
        
        if regex_pattern == "contract_value_regex":
            regex_pattern = contract_value_regex
            while i < len(sentences):
                if re.search(regex_pattern, sentences[i], re.VERBOSE | re.IGNORECASE):
                    print(f"Matched Sentence {i}: {sentences[i]}\n\n")
                    # Extract 1 sentence before, the matched sentence, and 1 sentence after
                    start = max(0, i - 2)
                    end = min(len(sentences), i + 2)  # Include the matched sentence and 1 after
                    selected_sentences = sentences[start:end]

                    # Limit the length of sentences before and after to 25 words
                    trimmed_sentences = []
                    for j, sent in enumerate(selected_sentences):
                        if sent not in seen_sentences:  # Avoid duplicate sentences
                            words = word_tokenize(sent)
                            if j == 0 and i > 0:  # Sentence before
                                trimmed_sentences.append(" ".join(words[-50:]))
                            elif j == len(selected_sentences) - 1 and i + 1 < len(sentences):  # Sentence after
                                trimmed_sentences.append(" ".join(words[:50]))
                            else:  # Matched sentence
                                trimmed_sentences.append(sent)
                            seen_sentences.add(sent)  # Mark sentence as seen

                    combined_text.extend(trimmed_sentences)
                    i = end  # Skip to the sentence after the extracted range to avoid overlap
                else:
                    i += 1

            # Combine all matched text into a single chunk
            if combined_text:
                filtered = [{"chunk_number": "combined", "text": " ".join(combined_text)}]
                self.logger.info(self._log_message(f"Filtered on line 563: {filtered}", "extract_regex_chunks_with_words"))
                return filtered[0]['text']
            return []

    @mlflow.trace(name="Metadata Extractor - Extract Meta Data Parallely")
    def extract_meta_data_parallely(self, file_id, file_name, file_type, user_id, org_id, retry_count, chunks):
        overall_start = time.perf_counter()
        process = psutil.Process()
        try:
            self.logger.info(self._log_message("Starting metadata extraction.", "extract_meta_data_parallely"))
            start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            status_start = time.perf_counter()
            

            set_llm_file_status(file_id, user_id, org_id, 3, True, retry_count, 1, "", start_datetime, "", False, False, self.in_queue, 25, self.logger)
            status_duration = time.perf_counter() - status_start
            
            custom_filter = {"file_id": {"$eq": file_id}}
            top_k = 10
            results = []
            
            def submit_with_context(executor, fn, *args, **kwargs):
                # Capture the current OpenTelemetry context (which includes the parent span)
                parent_ctx = ot_context.get_current()
                def wrapped():
                    token = ot_context.attach(parent_ctx)
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        ot_context.detach(token)
                return executor.submit(wrapped)
            
            
            @mlflow.trace(name="Metadata Extractor - Process Question")
            def process_question(question):
                try:
                    query = list(question.keys())[0]
                    retrieve_start = time.perf_counter()
                    
                    if query == "What is the title of the contract?":
                        custom_filter = {"file_id": {"$eq": file_id}, "page_no": {"$eq": 1}}
                    elif query == "What are the parties involved?":
                        custom_filter = {"file_id": {"$eq": file_id}, "page_no": {"$eq": 1}}
                    else:
                        custom_filter = {"file_id": {"$eq": file_id}}

                    context_chunks = get_context_from_pinecone(DOCUMENT_SUMMARY_AND_CHUNK_INDEX, PINECONE_API_KEY, custom_filter, top_k, query, file_id, user_id, org_id, self.logger)
                    retrieve_duration = time.perf_counter() - retrieve_start
                    
                    matches = context_chunks.get('matches', [])
                    if not matches:
                        return []
                    
                    context = [match['metadata']['text'] for match in matches]
                    # context = "/n here is more context:"+ first_page_context_chunks
                    # Log the query and the retrieved context
                    log_entry = {
                        "file_id": file_id,
                        "file_name": file_name,
                        "file_type": file_type,
                        "user_id": user_id,
                        "org_id": org_id,
                        "retry_count": retry_count,
                        "query": query,
                        "retrieved_context": context,
                        "retrieval_time": round(retrieve_duration, 2)
                    }
                    self.logger.info(self._log_message(f"METADATA QUERY AND CONTEXTS RETRIEVED: {orjson.dumps(log_entry).decode()}", "extract_meta_data_parallely"))
                    self.logger.debug(self._log_message(f"Context Chunks Retrieved: {len(context)}", "extract_meta_data_parallely"))
                    
                    process_start = time.perf_counter()
                    response = call_llm(question, context, file_id, user_id, org_id, self.logger)
                    process_duration = time.perf_counter() - process_start
                    
                    return response
                except Exception as e:
                    self.logger.error(self._log_message(f"Error processing question: {e}", "extract_meta_data_parallely"))
                    return []
            
            self.logger.info(self._log_message(f"Calling extract_regex_chunks_with_words", "extract_meta_data_parallely"))
            retrieved_chunks = self.extract_regex_chunks_with_words(chunks, "date_pattern")
            self.logger.info(self._log_message(f"Retrieved Chunks: {retrieved_chunks}", "extract_meta_data_parallely"))
            retrieved_chunks_jurisdiction = self.extract_regex_chunks_with_words(chunks, "jurisdiction_regex")
            self.logger.info(self._log_message(f"Retrieved Chunks Jurisdiction: {retrieved_chunks_jurisdiction}", "extract_meta_data_parallely"))
            retrieved_chunks_cv = self.extract_regex_chunks_with_words(chunks, "contract_value_regex")
            self.logger.info(self._log_message(f"Retrieved Chunks Contract Value: {retrieved_chunks_cv}", "extract_meta_data_parallely"))
            # Step 1: Extract dates using LLM
            date_extraction = llm_call_for_dates(retrieved_chunks, date_extraction_instructions, file_id, user_id, org_id, self.logger)
            jurisdiction_extraction = llm_call_for_jurisdiction(retrieved_chunks_jurisdiction, jurisdiction_instruction, file_id, user_id, org_id, self.logger)
            contract_value_extraction = llm_call_for_cv(retrieved_chunks_cv, contract_value_instructions, file_id, user_id, org_id, self.logger)


            # Step 2: Get keys with null or None values
            null_date_keys = [key for key, value in date_extraction.items() if value in ("null", None)]
            if jurisdiction_extraction.get("Jurisdiction") == "null" or jurisdiction_extraction.get("Jurisdiction") is None:
                null_date_keys.append("Jurisdiction")
            if contract_value_extraction.get("Contract Value") == "null" or contract_value_extraction.get("Contract Value") is None:
                null_date_keys.append("Contract Value")

            # Step 3: Collect relevant questions from metadata
            null_date_keys_questions = [
                data
                for date_key in null_date_keys
                for data in metadata_hybrid_queries
                if date_key in data
            ]

            # Optional debug print (replace with logger if needed)
            for q in null_date_keys_questions:
                for k, v in q.items():
                    print(f"Question.key: {k}")
                    print(f"Question.value: {v}")

            # Step 4: Process questions in parallel
            hybrid_dates = []
            with ThreadPoolExecutor() as executor:
                futures = {submit_with_context(executor, process_question, q): q for q in null_date_keys_questions}
                for future in as_completed(futures):
                    try:
                        hybrid_dates.append(future.result())
                    except Exception as e:
                        self.logger.error(self._log_message(f"Error collecting results: {e}", "extract_meta_data_parallely"))

            # Step 5: Merge new date values into the original result
            for result in hybrid_dates:
                for key, value in result.items():
                    if key in date_extraction:
                        date_extraction[key] = value
                    elif key == "Jurisdiction":
                        jurisdiction_extraction["Jurisdiction"] = value
                    elif key == "Contract Value":   
                        contract_value_extraction["Contract Value"] = value
        
            self.logger.info(self._log_message(f"Final Date Extraction Result: {date_extraction}", "extract_meta_data_parallely"))
            extraction_start = time.perf_counter()

            with ThreadPoolExecutor() as executor:
                # future_to_question = {executor.submit(process_question, question): question for question in METADATA_EXTRACTION_PROMPTS}
                
                future_to_question = {submit_with_context(executor, process_question, question): question for question in METADATA_EXTRACTION_PROMPTS}

                for future in as_completed(future_to_question):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        self.logger.error(self._log_message(f"Error collecting results: {e}", "extract_meta_data_parallely"))
            
            extraction_duration = time.perf_counter() - extraction_start
            
            self.logger.info(self._log_message(f"Metadata extraction results: {results}", "extract_meta_data_parallely"))
            
            expiry_date = None
            results.append(date_extraction)
            results.append(jurisdiction_extraction)
            results.append(contract_value_extraction)
            for result in results:
                if not result:
                    continue
                self.logger.info(self._log_message(f"Processing Result: {result} | Type - {type(result)}", "extract_meta_data_parallely"))
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key == "Expiration Date":
                            expiry_date = value
                            print(f"Expiration date: {expiry_date}")
                            break
            
            
            

            try:
                prompt_for_recurring_payment_due_date = {"What is the payment due date?": f"""Instructions:
                    Two parts, part A and part B:
                    Part A:
                        Determine if the contract supports recurring payments based on the given text.

                        Criteria:
                        1. Identify if the contract mentions payments on a recurring basis using keywords such as 'monthly', 'quarterly', 'annually' or phrases like:
                        - 'Payments due on the [specific date] of each month.'
                        - 'Quarterly payments due on [specific months].'
                        - 'Annual payment due on [specific date] every year.'
                        
                        2. Extract the contract expiry date.
                        3. Extract the payment due dates.
                        4. If the payment due dates extend beyond the contract expiry date, return 'false'.
                        5. If the contract explicitly states recurring payments and they fall within the contract period, return 'true'.
                        6. If no recurring payment terms are found, return 'false'.

                        Return only a single word: 'true' or 'false'.

                    Part B:    
                        Given the following relevant contract information, determine the next immediate payment due date based on the present date that is: {current_date}.

                        Consider the following scenarios:
                        1. **Fixed Payment Due Date:** If the contract specifies a one-time or fixed payment due date, return that date if it is in the future or have passed.
                        2. **Recurring Payment Due Dates:** If the contract specifies recurring payments (e.g., monthly, quarterly, annually), identify the next immediate due date considering the present date. The recurrence pattern can be in formats such as:
                        - "Payments due on the 15th of each month"
                        - "Quarterly payments due on tllm_call_for_dateshe first of January, April, July, and October"
                        - "Annual payment due on June 30 every year"

                        **Instructions:**  
                        - If no payment due date is found or it's unclear, return `'null'` as a string.
                        - The output should be a JSON-style object with the following structure:
                        {{"Payment Due Date": "YYYY-MM-DD"}}

                    OUTPUT:
                    {{
                    "flag": true/false,
                    "Payment Due Date": "YYYY-MM-DD"
                    }}
                    ###If Expiry date({expiry_date}) < 'Payment Due Date', then return 'flag' = false and 'Payment Due Date' = 'null' 
                    ###If Expiry date({expiry_date}) < "Current Date ({current_date})", then return "flag" = false and "Payment Due Date" = "null"
                    Now, analyze the following contract text and return the next immediate payment due date in the required format.

                    """ }
                # prompt_for_recurring_payment_due_date["What is the payment due date?"] = \
                #     prompt_for_recurring_payment_due_date["What is the payment due date?"].format(
                #         current_date=current_date, expiry_date=expiry_date
                #     )
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()  # Captures full error traceback

                self.logger.error(self._log_message(f"Error formatting prompt: {e}", "extract_meta_data_parallely"))
                # self.logger.error(f"Prompt before formatting: {prompt_for_recurring_payment_due_date.get('What is the payment due date?', 'Key not found')}")
                self.logger.error(f"Full Traceback:\n{error_details}")
                # raise
                # prompt_for_recurring_payment_due_date["What is the payment due date?"] = prompt_for_recurring_payment_due_date["What is the payment due date?"].format(current_date=current_date, expiry_date="null")
            
            recurring_start = time.perf_counter()
            
            response_for_recurring_payment = process_question(prompt_for_recurring_payment_due_date)
            self.logger.info(self._log_message(f"Recurring Payment Extraction Result: {response_for_recurring_payment} | Type - {type(response_for_recurring_payment)}", "extract_meta_data_parallely"))
            recurring_duration = time.perf_counter() - recurring_start

            if not response_for_recurring_payment:
                self.logger.error(self._log_message("Recurring Payment Extraction Failed", "extract_meta_data_parallely"))
            else:
                if isinstance(response_for_recurring_payment, dict):
                    payment_due_date = response_for_recurring_payment.get("Payment Due Date", None)
                    payment_due_date = None if payment_due_date == "null" else payment_due_date
                    flag_value = "Yes" if response_for_recurring_payment.get("flag", None) else "No"
                    flag_value, payment_due_date = payment_due_date_validatior(flag_value, payment_due_date, expiry_date, current_date, file_id, user_id, org_id, self.logger)

                results.extend([{'Payment Due Date': payment_due_date}, {'Has Recurring Payment': flag_value}])
            
            set_llm_file_status(file_id, user_id, org_id, 3, True, retry_count, 1, "", start_datetime, "", False, False, self.in_queue, 50, self.logger)

            @mlflow.trace(name="Metadata Extractor - Map Metadata")
            def map_metadata(data, dates_metadata, others_metadata):
                for entry in data:
                    self.logger.info(self._log_message(f"Processing Entry: {entry} | Type - {type(entry)}", "map_metadata"))
                    if isinstance(entry, dict):
                        for key, value in entry.items():
                            value = None if value == 'null' else value
                            for date_metadata in dates_metadata:
                                if key.strip().lower() == date_metadata["title"].strip().lower():
                                    date_metadata["value"] = value
                            for other_metadata in others_metadata:
                                if key.strip().lower() == other_metadata["title"].strip().lower():
                                    other_metadata["value"] = value

                return dates_metadata, others_metadata

            def update_contract_duration(dates_metadata, others_metadata):
                # Find the contract duration metadata item
                contract_duration_item = next((item for item in others_metadata if item["title"] == "Contract Duration"), None)
                
                # If contract duration is empty/None
                if contract_duration_item and contract_duration_item["value"] is None:
                    # Find effective date
                    effective_date_item = next((item for item in dates_metadata if item["title"] == "Effective Date"), None)
                    # Find expiration date
                    expiration_date_item = next((item for item in dates_metadata if item["title"] == "Expiration Date"), None)
                    # Find termination date
                    termination_date_item = next((item for item in dates_metadata if item["title"] == "Termination Date"), None)
                    
                    # Check if effective date exists and either expiration or termination date exists
                    if effective_date_item and effective_date_item["value"]:
                        end_date = None
                        
                        if expiration_date_item and expiration_date_item["value"]:
                            end_date = expiration_date_item["value"]
                        elif termination_date_item and termination_date_item["value"]:
                            end_date = termination_date_item["value"]
                        
                        if end_date:
                            contract_duration_item["value"] = f"from {effective_date_item['value']} to {end_date}"
                
                return dates_metadata, others_metadata

            default_dates_metadata = [
                {"title": "Effective Date", "value": None},
                {"title": "Term Date", "value": None},
                {"title": "Payment Due Date", "value": None},
                {"title": "Delivery Date", "value": None},
                {"title": "Termination Date", "value": None},
                {"title": "Renewal Date", "value": None},
                {"title": "Expiration Date", "value": None}
            ]

            default_others_metadata = [
                {"title": "Title of the Contract", "value": None},
                {"title": "Scope of Work", "value": None},
                {"title": "Parties Involved", "value": None},
                {"title": "Contract Type", "value": None},
                {"title": "File Type", "value": "Contract"},
                {"title": "Jurisdiction", "value": None},
                {"title": "Version Control", "value": None},
                {"title": "Contract Duration", "value": None},
                {"title": "Contract Value", "value": None},
                {"title": "Risk Mitigation Score", "value": None},
                {"title": "Has Recurring Payment", "value": None}
            ]

            mapping_start = time.perf_counter()
            dates_metadata, others_metadata = map_metadata(results, default_dates_metadata, default_others_metadata)
            # Add the validation for contract duration
            dates_metadata, others_metadata = update_contract_duration(dates_metadata, others_metadata)

            mapping_duration = time.perf_counter() - mapping_start

            metadata = {"metadata": {"dates": dates_metadata, "others": others_metadata}}

          
            set_llm_file_status(file_id, user_id, org_id, 3, True, retry_count, 1, "", start_datetime, "", False, False, self.in_queue, 75, self.logger)
            
            set_meta_start = time.perf_counter()
            set_meta_data(file_id, user_id, org_id, 3, start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 3, False, "", retry_count, metadata, self.in_queue, 100, self.logger)
            set_meta_duration = time.perf_counter() - set_meta_start
            
            vector_start = time.perf_counter()
            self.metadata_vector_handler.process_contract_template(metadata, file_id, file_name, file_type, user_id, org_id, len(chunks))
            vector_duration = time.perf_counter() - vector_start
            
            status_end_start = time.perf_counter()
            

            set_llm_file_status(file_id, user_id, org_id, 3, True, retry_count, 3, "", start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), False, False, self.in_queue, 100, self.logger)
            status_end_duration = time.perf_counter() - status_end_start
            
            overall_duration = time.perf_counter() - overall_start
            memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
            cpu_usage = process.cpu_percent(interval=0.1)
            
            log_data = {
                "file_id": file_id,
                "file_name": file_name,
                "file_type": file_type,
                "user_id": user_id,
                "org_id": org_id,
                "retry_count": retry_count,
                "metadata": metadata,
                "memory_usage": round(memory_usage, 2),
                "cpu_usage": round(cpu_usage, 2),
                "time_taken": {
                    "llm_status_initiate_3_time": round(status_duration, 2),
                    "metadata_extraction_time": round(extraction_duration, 2),
                    "recurring_payment_extraction_time": round(recurring_duration, 2),
                    "metadata_status_completion_time": round(set_meta_duration, 2),
                    "metadata_vector_upsertion_time": round(vector_duration, 2),
                    "llm_status_completion_3_time": round(status_end_duration, 2),
                    "total_metadata_execution_time": round(overall_duration, 2)
                }
            }
            self.logger.info(self._log_message(f"METADATA EXTRACTION PROCESS SUMMARY: {orjson.dumps(log_data).decode()}", "extract_meta_data_parallely"))
            return metadata
        except Exception as e:
            error_message = f"Error during metadata extraction: {e}"
            self.logger.error(self._log_message(error_message, "extract_meta_data_parallely"))
            
            set_llm_file_status(file_id, user_id, org_id, 3, True, retry_count, 2, error_message, start_datetime, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), False, False, self.in_queue, 100, self.logger)
            raise

        finally:
            elapsed_time = time.perf_counter() - overall_start
            self.logger.info(self._log_message(f"Metadata extraction completed in {elapsed_time:.2f} seconds.", "extract_meta_data_parallely"))
