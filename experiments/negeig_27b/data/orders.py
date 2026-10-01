"""orders: a customer-support desk. Orders (#4817) hold products, have a status and a shipping city.
Questions: status of order #N, items in order #N (sorted, ", "-joined, or none), shipping city of order #N.

Events:
  place    a new order is placed with items and a city                        (new container)
  pay      placed -> paid (only orders that hold an item)                     (overwrite)
  ship     paid -> shipped (only orders that hold an item)
  deliver  shipped -> delivered
  return   delivered -> returned
  cancel   placed or paid -> cancelled
  address  the shipping city of a placed or paid order is overwritten
  move     an item moves from one placed/paid order to another                (move between containers)
  swap     two placed/paid orders trade one item each                         (transposition between containers)
  add      an item is added to a placed/paid order
  remove   an item is removed from a placed/paid order
  noop     the customer asks for an update, the agent adds a note, the address is confirmed   (no change)
  declined a cancel/change request on an order that is not placed or paid                  (no change)
English and Hebrew. Order IDs stay Latin; products and cities are Hebrew in Hebrew sessions. The Hebrew events
use the agent ("הנציג", masculine role noun) or an order (feminine) as the subject, so no item gender is needed.
"""

from __future__ import annotations

import random
import re

from common import Session, split_pool

STATUSES = ["placed", "paid", "shipped", "delivered", "cancelled", "returned"]
EDITABLE = ("placed", "paid")
MAX_ITEMS = 6
STATUS_WORD = {
    "en": {"placed": "placed", "paid": "paid", "shipped": "shipped", "delivered": "delivered",
           "cancelled": "cancelled", "returned": "returned"},
    "he": {"placed": "התקבלה", "paid": "שולמה", "shipped": "נשלחה", "delivered": "נמסרה",
           "cancelled": "בוטלה", "returned": "הוחזרה"},
}
NONE_WORD = {"en": "none", "he": "אין"}

def _order_nums() -> list[int]:
    nums = list(range(1000, 10000))
    random.Random(20260930).shuffle(nums)  # fixed shuffle: the train/eval halves are not digit-parity classes
    return nums


ORDER_NUMS = _order_nums()  # split_pool halves: eval order numbers are never seen in training, all digits occur in both

PRODUCTS = {
    "en": [
        "wireless mouse", "mechanical keyboard", "desk lamp", "laptop stand", "usb hub", "webcam", "monitor arm",
        "phone case", "screen cleaner", "notebook", "travel mug", "water bottle", "backpack", "yoga mat",
        "running shoes", "wool socks", "rain jacket", "sun hat", "bike helmet", "camping stove", "sleeping bag",
        "hiking boots", "board game", "jigsaw puzzle", "coffee grinder", "tea kettle", "cutting board",
        "chef knife", "frying pan", "bath towel", "table runner", "wall clock", "picture frame", "candle set",
        "plant pot", "garden hose", "tool set", "paint roller", "desk organizer", "bluetooth speaker",
    ],
    "he": [
        "העכבר האלחוטי", "המקלדת המכנית", "מנורת השולחן", "מעמד המחשב", "מצלמת הרשת", "זרוע המסך",
        "נרתיק הטלפון", "תרסיס הניקוי", "המחברת", "כוס הנסיעה", "בקבוק המים", "התרמיל", "מזרן היוגה",
        "נעלי הריצה", "גרבי הצמר", "מעיל הגשם", "כובע השמש", "קסדת האופניים", "תנור הקמפינג", "שק השינה",
        "נעלי הטיולים", "משחק הקופסה", "הפאזל", "מטחנת הקפה", "קומקום התה", "קרש החיתוך", "סכין השף",
        "המחבת", "מגבת הרחצה", "מפת השולחן", "שעון הקיר", "מסגרת התמונה", "סט הנרות", "עציץ הפרחים",
        "צינור ההשקיה", "ערכת הכלים", "גליל הצבע", "מארגן השולחן", "הרמקול האלחוטי", "מטען הנסיעה",
    ],
}

CITIES = {
    "en": [
        "Lisbon", "Denver", "Leeds", "Porto", "Turin", "Austin", "Bergen", "Perth", "Geneva", "Dublin", "Seattle",
        "Utrecht", "Ghent", "Krakow", "Malaga", "Bristol", "Oslo", "Tampa", "Graz", "Lille", "Reno", "Tartu",
        "Basel", "Cork",
    ],
    "he": [
        "חיפה", "תל אביב", "ירושלים", "באר שבע", "נתניה", "אשדוד", "אילת", "טבריה", "ראשון לציון", "פתח תקווה",
        "חולון", "רחובות", "הרצליה", "כפר סבא", "מודיעין", "נצרת", "אשקלון", "חדרה", "עפולה", "עכו", "צפת",
        "לוד", "רמת גן", "בת ים", "כרמיאל", "קריית שמונה", "נהריה", "דימונה",
    ],
}

TEMPLATES = {
    "en": {
        "place": ["Customer places new order #{o}, shipping to {c}, with items: {items}."],
        "pay": ["Order #{o} is paid.", "Payment received for order #{o}."],
        "ship": ["Order #{o} is shipped.", "Order #{o} is handed to the courier."],
        "deliver": ["Order #{o} is delivered.", "Delivery of order #{o} is confirmed."],
        "cancel": ["Order #{o} is cancelled.", "Customer cancels order #{o}."],
        "return": ["Order #{o} is returned.", "Customer returns order #{o}."],
        "address": ["Order #{o} now ships to {c}.", "Shipping city of order #{o} is changed to {c}."],
        "move": ["Agent moves the {x} from order #{a} to order #{b}."],
        "swap": ["Agent swaps the {x} in order #{a} with the {y} in order #{b}."],
        "add": ["Agent adds the {x} to order #{o}."],
        "remove": ["Agent removes the {x} from order #{o}."],
        "noop_update": ["Customer asks for an update on order #{o}."],
        "noop_note": ["Agent adds a note to order #{o}."],
        "noop_confirm": ["Customer confirms the address of order #{o}."],
        "decl_cancel": ["Request to cancel order #{o} is declined."],
        "decl_change": ["Request to change order #{o} is declined."],
        "q_status": ["What is the status of order #{o}?"],
        "q_items": ["Which items are in order #{o}?"],
        "q_city": ["What is the shipping city of order #{o}?"],
        "init": ["Order #{o}: {st}, shipping to {c}, items: {items}."],
    },
    "he": {
        "place": ["התקבלה הזמנה חדשה #{o} למשלוח ל{c}, עם הפריטים: {items}."],
        "pay": ["הזמנה #{o} שולמה.", "התקבל תשלום עבור הזמנה #{o}."],
        "ship": ["הזמנה #{o} נשלחה.", "הזמנה #{o} יצאה למשלוח."],
        "deliver": ["הזמנה #{o} נמסרה.", "המסירה של הזמנה #{o} אושרה."],
        "cancel": ["הזמנה #{o} בוטלה.", "הלקוח מבטל את הזמנה #{o}."],
        "return": ["הזמנה #{o} הוחזרה.", "הלקוח מחזיר את הזמנה #{o}."],
        "address": ["הזמנה #{o} תישלח מעתה ל{c}.", "עיר המשלוח של הזמנה #{o} שונתה ל{c}."],
        "move": ["הנציג מעביר את {x} מהזמנה #{a} להזמנה #{b}."],
        "swap": ["הנציג מחליף בין {x} בהזמנה #{a} לבין {y} בהזמנה #{b}."],
        "add": ["הנציג מוסיף את {x} להזמנה #{o}."],
        "remove": ["הנציג מסיר את {x} מהזמנה #{o}."],
        "noop_update": ["הלקוח מבקש עדכון על הזמנה #{o}."],
        "noop_note": ["הנציג מוסיף הערה להזמנה #{o}."],
        "noop_confirm": ["הלקוח מאשר את הכתובת של הזמנה #{o}."],
        "decl_cancel": ["הבקשה לבטל את הזמנה #{o} נדחתה."],
        "decl_change": ["הבקשה לשנות את הזמנה #{o} נדחתה."],
        "q_status": ["מה הסטטוס של הזמנה #{o}?"],
        "q_items": ["אילו פריטים יש בהזמנה #{o}?"],
        "q_city": ["מהי עיר המשלוח של הזמנה #{o}?"],
        "init": ["הזמנה #{o}: {st}, משלוח ל{c}, פריטים: {items}."],
    },
}


class Sim:
    SYSTEM = {
        "en": "You are a customer-support desk assistant tracking orders. You get the initial state of the orders, "
              "then events in order. After each batch, answer the question with only the value, nothing else. "
              "Status words: placed, paid, shipped, delivered, cancelled, returned. A new order starts as placed. "
              "An order goes from placed to paid, then shipped, then delivered, then returned, and it can be cancelled while it is placed or "
              "paid. Items and the shipping city can only change while an order is placed or paid; a request on "
              "any other order is declined and changes nothing. Cancelled and returned orders keep their items. "
              "For items, answer the product names sorted alphabetically and joined by \", \", or none if the "
              "order has no items. For the shipping city, answer the city name.",
        "he": "את/ה עוזר/ת במוקד תמיכה ועוקב/ת אחרי הזמנות. מקבלים מצב התחלתי של ההזמנות ואחריו אירועים לפי הסדר. "
              "אחרי כל קבוצת אירועים עונים על השאלה בערך בלבד, בלי מילים נוספות. מילות הסטטוס: התקבלה, שולמה, "
              "נשלחה, נמסרה, בוטלה, הוחזרה. הזמנה חדשה מתחילה במצב התקבלה. הזמנה עוברת מהתקבלה לשולמה, אחר כך לנשלחה, לנמסרה ולהוחזרה, ואפשר "
              "לבטל אותה כל עוד היא במצב התקבלה או שולמה. אפשר לשנות פריטים ועיר משלוח רק בהזמנה במצב התקבלה "
              "או שולמה; בקשה לשנות הזמנה אחרת נדחית ולא משנה דבר. הזמנות שבוטלו או הוחזרו שומרות את הפריטים "
              "שלהן. בשאלה על פריטים עונים בשמות המוצרים לפי סדר האלף-בית כפי שהם כתובים (כולל ה' הידיעה), "
              "מופרדים בפסיק וברווח (\", \"), או אין אם בהזמנה אין פריטים. בשאלה על עיר המשלוח עונים בשם העיר.",
    }

    def __init__(self, rng, lang, split):
        self.rng, self.lang = rng, lang
        self.products = rng.sample(split_pool(PRODUCTS[lang], split), rng.randint(10, 16))
        self.cities = rng.sample(split_pool(CITIES[lang], split), rng.randint(6, 10))
        self.id_pool = split_pool(ORDER_NUMS, split)
        self.status: dict[int, str] = {}
        self.city: dict[int, str] = {}
        self.items: dict[int, set] = {}
        self.n_changes: dict[int, int] = {}     # real state changes since the start (0 = untouched)
        self.last_touch: dict[int, int] = {}    # step of the last real change
        self.last_attr: dict[int, str] = {}     # attribute the last change hit: status / items / city
        self.noop_touch: dict[int, int] = {}    # step of the last no-change mention
        self.step = 0
        self._init_lines: list[str] = []
        self.initial_ids: list[int] = []
        for i in range(rng.randint(5, 8)):
            st = rng.choice(EDITABLE) if i < 2 else rng.choices(STATUSES, weights=[30, 28, 20, 12, 6, 4])[0]
            o, shown = self._open(st, rng.randint(1, 4), 0)
            self._init_lines.append(self._t("init", o=o, st=STATUS_WORD[lang][st], c=self.city[o],
                                            items=", ".join(shown)))
            self.initial_ids.append(o)

    # ---------- helpers
    def _t(self, key, **kw):
        return self.rng.choice(TEMPLATES[self.lang][key]).format(**kw)

    def _open(self, st, n, changes):
        rng = self.rng
        while True:
            o = rng.choice(self.id_pool)
            if o not in self.status:
                break
        shown = rng.sample(self.products, n)  # rendered order: random, not sorted
        self.status[o], self.city[o], self.items[o] = st, rng.choice(self.cities), set(shown)
        self.n_changes[o] = changes
        self.last_touch[o] = self.step if changes else -99
        self.last_attr[o] = rng.choice(["status", "items", "city"]) if changes else "status"
        return o, shown

    def _touch(self, o, attr):
        self.n_changes[o] += 1
        self.last_touch[o] = self.step
        self.last_attr[o] = attr

    def _pick(self, cands):
        """Half the time prefer an order touched in the last 10 events, so orders change several times."""
        rec = [o for o in cands if self.last_touch[o] >= self.step - 10]
        if rec and self.rng.random() < 0.5:
            return self.rng.choice(rec)
        return self.rng.choice(cands)

    # ---------- Sim interface
    def initial(self):
        return list(self._init_lines)

    def event(self):
        rng, st = self.rng, self.status
        self.step += 1
        E = [o for o in st if st[o] in EDITABLE]
        P = [o for o in st if st[o] == "placed" and self.items[o]]  # an empty order is never paid or shipped
        D = [o for o in st if st[o] == "paid" and self.items[o]]
        S = [o for o in st if st[o] == "shipped"]
        V = [o for o in st if st[o] == "delivered"]
        N = [o for o in st if st[o] not in EDITABLE]
        live = len(E) + len(S)
        opts = ["noop"] * 4 + ["place"] * ((8 if live < 6 else 3 if live < 9 else 1) + (6 if len(E) < 3 else 0))
        if P:
            opts += ["pay"] * 4
        if D:
            opts += ["ship"] * 3
        if S:
            opts += ["deliver"] * 3
        if V:
            opts += ["return"] * 1
        if E:
            opts += ["cancel"] * 1 + ["address"] * 3 + ["add"] * 3
            if any(self.items[o] for o in E):
                opts += ["remove"] * 3
        if len(E) >= 2:
            opts += ["move"] * 6 + ["swap"] * 6
        if N:
            opts += ["declined"] * 2
        kind = rng.choice(opts)
        pools = {"pay": P, "ship": D, "deliver": S, "return": V, "cancel": E, "address": E}
        if kind in ("pay", "ship", "deliver", "return", "cancel"):
            o = self._pick(pools[kind])
            st[o] = {"pay": "paid", "ship": "shipped", "deliver": "delivered", "return": "returned",
                     "cancel": "cancelled"}[kind]
            self._touch(o, "status")
            return self._t(kind, o=o)
        if kind == "place":
            n = rng.choices([1, 2, 3, 4], weights=[3, 4, 3, 2])[0]
            o, shown = self._open("placed", n, 1)
            return self._t("place", o=o, c=self.city[o], items=", ".join(shown))
        if kind == "address":
            o = self._pick(E)
            self.city[o] = rng.choice([c for c in self.cities if c != self.city[o]])
            self._touch(o, "city")
            return self._t("address", o=o, c=self.city[o])
        if kind == "add":
            cands = [o for o in E if len(self.items[o]) < MAX_ITEMS]
            if cands:
                o = self._pick(cands)
                x = rng.choice([p for p in self.products if p not in self.items[o]])
                self.items[o].add(x)
                self._touch(o, "items")
                return self._t("add", o=o, x=x)
        if kind == "remove":
            cands = [o for o in E if self.items[o]]
            if cands:
                o = self._pick(cands)
                x = rng.choice(sorted(self.items[o]))
                self.items[o].discard(x)
                self._touch(o, "items")
                return self._t("remove", o=o, x=x)
        if kind == "move":
            for _ in range(8):
                a = self._pick(E)
                if not self.items[a]:
                    continue
                x = rng.choice(sorted(self.items[a]))
                bs = [b for b in E if b != a and x not in self.items[b] and len(self.items[b]) < MAX_ITEMS]
                if bs:
                    b = self._pick(bs)
                    self.items[a].discard(x)
                    self.items[b].add(x)
                    self._touch(a, "items")
                    self._touch(b, "items")
                    return self._t("move", x=x, a=a, b=b)
        if kind == "swap":
            for _ in range(8):
                a = self._pick(E)
                b = self._pick([o for o in E if o != a])
                pairs = [(x, y) for x in sorted(self.items[a]) for y in sorted(self.items[b])
                         if x not in self.items[b] and y not in self.items[a]]
                if pairs:
                    x, y = rng.choice(pairs)
                    self.items[a].discard(x)
                    self.items[b].discard(y)
                    self.items[a].add(y)
                    self.items[b].add(x)
                    self._touch(a, "items")
                    self._touch(b, "items")
                    return self._t("swap", x=x, y=y, a=a, b=b)
        if kind == "declined" and N:
            o = self._pick(N)
            self.noop_touch[o] = self.step
            return self._t(rng.choice(["decl_cancel", "decl_change"]), o=o)
        o = self._pick(list(st))  # no-op distractor (also the fallback when a move/swap/add found no legal target)
        self.noop_touch[o] = self.step
        return self._t(rng.choice(["noop_update", "noop_note", "noop_confirm"]), o=o)

    def ask(self):
        rng, st = self.rng, self.status
        orders = list(st)
        un = [o for o in self.initial_ids if self.n_changes[o] == 0]
        nz = [o for o in orders if self.noop_touch.get(o, -99) >= self.step - 10]
        multi = [o for o in orders if self.n_changes[o] >= 3]
        recent = [o for o in orders if self.last_touch[o] >= self.step - 12]
        r = rng.random()
        cls = "random"
        if r < 0.06 and un:
            cls, o = "untouched", rng.choice(un)
        elif r < 0.16 and nz:
            cls, o = "noop", rng.choice(nz)
        elif r < 0.36 and multi:
            cls, o = "multi", rng.choice(multi)
        elif r < 0.92 and recent:
            cls = "recent"
            o = rng.choices(recent, weights=[13 - (self.step - self.last_touch[x]) for x in recent])[0]
        else:
            o = rng.choice(orders)
        if cls in ("recent", "multi") and rng.random() < 0.65:
            attr = self.last_attr[o]
        else:
            attr = rng.choices(["status", "items", "city"], weights=[4, 4, 2])[0]
        if attr == "status":
            a = STATUS_WORD[self.lang][st[o]]
        elif attr == "items":
            a = ", ".join(sorted(self.items[o])) or NONE_WORD[self.lang]
        else:
            a = self.city[o]
        return self._t("q_" + attr, o=o), a, {"order": o, "attr": attr, "cls": cls}


# ---------- independent checker: recompute every answer from the rendered text only
def replay(s: Session) -> list[str]:
    lang = s.lang
    he = lang == "he"
    prods = set(PRODUCTS[lang])
    cities = set(CITIES[lang])
    word_status = {w: k for k, w in STATUS_WORD[lang].items()}
    alt_p = "|".join(re.escape(p) for p in sorted(prods, key=len, reverse=True))
    alt_c = "|".join(re.escape(c) for c in sorted(cities, key=len, reverse=True))
    alt_w = "|".join(re.escape(w) for w in word_status)
    N = r"#(\d{4})"
    if he:
        ev = {
            "place": rf"^התקבלה הזמנה חדשה {N} למשלוח ל({alt_c}), עם הפריטים: (.+)\.$",
            "pay": rf"^הזמנה {N} שולמה\.$|^התקבל תשלום עבור הזמנה {N}\.$",
            "ship": rf"^הזמנה {N} נשלחה\.$|^הזמנה {N} יצאה למשלוח\.$",
            "deliver": rf"^הזמנה {N} נמסרה\.$|^המסירה של הזמנה {N} אושרה\.$",
            "cancel": rf"^הזמנה {N} בוטלה\.$|^הלקוח מבטל את הזמנה {N}\.$",
            "return": rf"^הזמנה {N} הוחזרה\.$|^הלקוח מחזיר את הזמנה {N}\.$",
            "address": rf"^הזמנה {N} תישלח מעתה ל({alt_c})\.$|^עיר המשלוח של הזמנה {N} שונתה ל({alt_c})\.$",
            "move": rf"^הנציג מעביר את ({alt_p}) מהזמנה {N} להזמנה {N}\.$",
            "swap": rf"^הנציג מחליף בין ({alt_p}) בהזמנה {N} לבין ({alt_p}) בהזמנה {N}\.$",
            "add": rf"^הנציג מוסיף את ({alt_p}) להזמנה {N}\.$",
            "remove": rf"^הנציג מסיר את ({alt_p}) מהזמנה {N}\.$",
            "noop": rf"^הלקוח מבקש עדכון על הזמנה {N}\.$|^הנציג מוסיף הערה להזמנה {N}\.$"
                    rf"|^הלקוח מאשר את הכתובת של הזמנה {N}\.$",
            "decl": rf"^הבקשה לבטל את הזמנה {N} נדחתה\.$|^הבקשה לשנות את הזמנה {N} נדחתה\.$",
        }
        init = rf"^הזמנה {N}: ({alt_w}), משלוח ל({alt_c}), פריטים: (.+)\.$"
        q = {"status": rf"^מה הסטטוס של הזמנה {N}\?$", "items": rf"^אילו פריטים יש בהזמנה {N}\?$",
             "city": rf"^מהי עיר המשלוח של הזמנה {N}\?$"}
        none = "אין"
    else:
        ev = {
            "place": rf"^Customer places new order {N}, shipping to ({alt_c}), with items: (.+)\.$",
            "pay": rf"^Order {N} is paid\.$|^Payment received for order {N}\.$",
            "ship": rf"^Order {N} is shipped\.$|^Order {N} is handed to the courier\.$",
            "deliver": rf"^Order {N} is delivered\.$|^Delivery of order {N} is confirmed\.$",
            "cancel": rf"^Order {N} is cancelled\.$|^Customer cancels order {N}\.$",
            "return": rf"^Order {N} is returned\.$|^Customer returns order {N}\.$",
            "address": rf"^Order {N} now ships to ({alt_c})\.$|^Shipping city of order {N} is changed to ({alt_c})\.$",
            "move": rf"^Agent moves the ({alt_p}) from order {N} to order {N}\.$",
            "swap": rf"^Agent swaps the ({alt_p}) in order {N} with the ({alt_p}) in order {N}\.$",
            "add": rf"^Agent adds the ({alt_p}) to order {N}\.$",
            "remove": rf"^Agent removes the ({alt_p}) from order {N}\.$",
            "noop": rf"^Customer asks for an update on order {N}\.$|^Agent adds a note to order {N}\.$"
                    rf"|^Customer confirms the address of order {N}\.$",
            "decl": rf"^Request to cancel order {N} is declined\.$|^Request to change order {N} is declined\.$",
        }
        init = rf"^Order {N}: ({alt_w}), shipping to ({alt_c}), items: (.+)\.$"
        q = {"status": rf"^What is the status of order {N}\?$", "items": rf"^Which items are in order {N}\?$",
             "city": rf"^What is the shipping city of order {N}\?$"}
        none = "none"
    ev = {k: re.compile(v) for k, v in ev.items()}
    init = re.compile(init)
    q = {k: re.compile(v) for k, v in q.items()}

    status: dict[str, str] = {}
    city: dict[str, str] = {}
    items: dict[str, set] = {}

    def parse_items(text):
        parts = text.split(", ")
        assert parts and all(p in prods for p in parts) and len(set(parts)) == len(parts), text
        return set(parts)

    def one(m):  # the single order number a no-list pattern captured (alternatives leave the others None)
        g = [x for x in m.groups() if x is not None]
        assert len(g) == 1, g
        return g[0]

    def need(o, allowed, line):
        assert o in status, f"unknown order in {line!r}"
        assert status[o] in allowed, f"order {o} is {status[o]}, invalid for {line!r}"

    for line in s.initial:
        m = init.match(line)
        assert m, f"unparsed initial line: {line!r}"
        o, w, c, lst = m.groups()
        assert o not in status, line
        status[o], city[o], items[o] = word_status[w], c, parse_items(lst)
        assert items[o], line

    answers = []
    for t in s.turns:
        for line in t.events:
            for kind, rx in ev.items():
                m = rx.match(line)
                if m:
                    break
            else:
                raise AssertionError(f"unparsed event: {line!r}")
            if kind == "place":
                o, c, lst = m.groups()
                assert o not in status, line
                status[o], city[o], items[o] = "placed", c, parse_items(lst)
                assert 0 < len(items[o]) <= MAX_ITEMS, line
            elif kind in ("pay", "ship", "deliver", "return", "cancel"):
                o = one(m)
                frm, to = {"pay": (("placed",), "paid"), "ship": (("paid",), "shipped"),
                           "deliver": (("shipped",), "delivered"), "return": (("delivered",), "returned"),
                           "cancel": (("placed", "paid"), "cancelled")}[kind]
                need(o, frm, line)
                status[o] = to
            elif kind == "address":
                g = [x for x in m.groups() if x is not None]
                assert len(g) == 2, g
                o, c = g
                need(o, ("placed", "paid"), line)
                assert c != city[o], line
                city[o] = c
            elif kind == "move":
                x, a, b = m.groups()
                need(a, ("placed", "paid"), line)
                need(b, ("placed", "paid"), line)
                assert a != b and x in items[a] and x not in items[b], line
                items[a].remove(x)
                items[b].add(x)
            elif kind == "swap":
                x, a, y, b = m.groups()
                need(a, ("placed", "paid"), line)
                need(b, ("placed", "paid"), line)
                assert a != b and x in items[a] and y in items[b], line
                assert x not in items[b] and y not in items[a], line
                items[a].remove(x)
                items[b].remove(y)
                items[a].add(y)
                items[b].add(x)
            elif kind == "add":
                x, o = m.groups()
                need(o, ("placed", "paid"), line)
                assert x not in items[o] and len(items[o]) < MAX_ITEMS, line
                items[o].add(x)
            elif kind == "remove":
                x, o = m.groups()
                need(o, ("placed", "paid"), line)
                assert x in items[o], line
                items[o].remove(x)
            elif kind == "noop":
                assert one(m) in status, line
            else:  # declined: only ever raised on an order the request could not apply to
                o = one(m)
                need(o, ("shipped", "delivered", "cancelled", "returned"), line)
        for attr, rx in q.items():
            m = rx.match(t.question)
            if m:
                o = m.group(1)
                assert o in status, t.question
                answers.append(word_status_rev(status[o], lang) if attr == "status"
                               else (", ".join(sorted(items[o])) or none) if attr == "items" else city[o])
                break
        else:
            raise AssertionError(f"unparsed question: {t.question!r}")
    return answers


def word_status_rev(st: str, lang: str) -> str:
    return STATUS_WORD[lang][st]
