"""Conservative count parsing without rewriting identity, dates or the transcript."""
import re
WORDS={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'a couple':2,'couple':2,'a pair':2,'pair':2,'half a dozen':6,'a dozen':12,'dozen':12,'double':2,'triple':3,'a':1,'an':1}
COUNT=r'(?:half a dozen|a couple|a pair|a dozen|'+'|'.join(k for k in WORDS if ' ' not in k)+r'|\d{1,2})'
def number(value):return int(value) if value.isdigit() else WORDS[value.casefold()]
def party_count(text,current=None):
 text=' '.join(text.casefold().split())
 text=re.sub(r'\b(?:a|an)\s+(?=(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b)', '', text)
 relative=re.search(r'\b('+COUNT+r')\s+(more|additional|fewer|less)\s+(?:guests?|people|persons?|friends?)\b',text)
 if relative and current is not None:
  delta=number(relative[1]);return current+delta if relative[2] in ('more','additional') else current-delta
 m=re.search(r'\b(?:table for|party of)\s+('+COUNT+r')(?:\s+of)?\b',text)
 if m:return number(m[1])
 m=re.search(r"\b(?:make (?:that|it)|change (?:that|it) to|we(?: will|'ll) be)\s+("+COUNT+r')(?:\s+of)?(?:\s+(?:guests?|people|persons?))\b',text)
 if m:return number(m[1])
 m=re.search(r'\b('+COUNT+r')(?:\s+of)?\s+(?:guests?|people|persons?)\b',text)
 return number(m[1]) if m else None
def item_count(text,name,current=None):
 words=re.findall(r'[a-z0-9]+',name.casefold())
 if not words:return None
 last=words[-1]
 tail=re.escape(last[:-1])+r's?' if last.endswith('s') and not last.endswith('ss') else re.escape(last)+r's?'
 item=r'\s+'.join([*(re.escape(w) for w in words[:-1]),tail])
 matches=list(re.finditer(r'\b('+COUNT+r')(?:\s+(?:order|portion|serving)s?\s+of)?(?:\s+of)?\s+(?:the\s+)?(?:(more|additional)\s+)?'+item+r'\b',text.casefold()))
 if len(matches)!=1:return None
 m=matches[0]
 count=number(m[1])
 if m[2]:return None if current is None else current+count
 return count
def wants_review(text):
 return bool(re.search(r"\b(?:read.?back|read (?:it|that|the order) back|summary|review|that(?:'s| is) (?:all|everything)|ready to (?:order|confirm)|place (?:the|my) order)\b",text,re.I))
