"""Entity pools for the S1 generators. Train and eval draw disjoint halves (common.split_pool).

Hebrew people carry a grammatical gender ("m"/"f") for verb agreement. Hebrew places carry the forms the
templates need: definite, "in", "to" and "from" (the article merges with the prepositions ב and ל).
"""

PEOPLE_EN = [
    "Omer", "Dana", "Noa", "Yael", "Ariel", "Maya", "Liam", "Emma", "Noah", "Olivia", "Ethan", "Ava", "Lucas",
    "Mia", "Mason", "Sofia", "Logan", "Isabella", "James", "Amelia", "Elijah", "Harper", "Aiden", "Evelyn",
    "Carter", "Abigail", "Owen", "Ella", "Wyatt", "Scarlett", "Jack", "Grace", "Levi", "Chloe", "Henry",
    "Victoria", "Sebastian", "Riley", "Mateo", "Aria", "Julian", "Lily", "Leo", "Aurora", "Hudson", "Zoey",
    "Isaac", "Nora", "Jayden", "Hannah", "Gabriel", "Layla", "Anthony", "Stella", "Dylan", "Leah", "Asher",
    "Hazel", "Thomas", "Violet", "Charles", "Lucy", "Caleb", "Anna", "Josiah", "Samantha", "Christopher",
    "Caroline", "Andrew", "Genesis", "Theodore", "Aaliyah", "Jaxon", "Kennedy", "Joshua", "Kinsley", "Nathan",
    "Allison", "Ryan", "Madison", "Adrian", "Sarah", "Christian", "Madelyn", "Miles", "Adeline", "Eli", "Alexa",
    "Nolan", "Ariana", "Aaron", "Elena", "Cameron", "Gabriella", "Connor", "Naomi", "Easton", "Alice",
    "Jeremiah", "Sadie", "Ezekiel", "Hailey", "Colton", "Eva", "Jordan", "Emilia", "Robert", "Autumn",
    "Angel", "Quinn", "Greyson", "Nevaeh", "Cooper", "Piper", "Austin", "Ruby", "Declan", "Serenity",
    "Roman", "Willow", "Everett", "Everly", "Xavier", "Cora", "Axel", "Kaylee", "Kai", "Lydia",
]

PEOPLE_HE = [
    ("עומר", "m"), ("דנה", "f"), ("נועה", "f"), ("יעל", "f"), ("אריאל", "m"), ("מאיה", "f"), ("איתי", "m"),
    ("שירה", "f"), ("יונתן", "m"), ("תמר", "f"), ("אורי", "m"), ("רוני", "f"), ("דניאל", "m"), ("מיכל", "f"),
    ("אביב", "m"), ("הילה", "f"), ("גיל", "m"), ("ליאת", "f"), ("עידו", "m"), ("קרן", "f"), ("נדב", "m"),
    ("אורית", "f"), ("אלון", "m"), ("טליה", "f"), ("ליאור", "m"), ("אפרת", "f"), ("רועי", "m"), ("סיון", "f"),
    ("עמית", "m"), ("ענבל", "f"), ("יובל", "m"), ("הדס", "f"), ("אסף", "m"), ("נטע", "f"), ("בועז", "m"),
    ("לירון", "f"), ("שחר", "m"), ("אביגיל", "f"), ("תומר", "m"), ("רחל", "f"), ("אייל", "m"), ("שני", "f"),
    ("דור", "m"), ("מורן", "f"), ("ערן", "m"), ("גלית", "f"), ("יואב", "m"), ("אסנת", "f"), ("אלעד", "m"),
    ("רינת", "f"), ("ניר", "m"), ("אילנה", "f"), ("צחי", "m"), ("ורד", "f"), ("משה", "m"), ("שרון", "f"),
    ("אבי", "m"), ("לימור", "f"), ("רון", "m"), ("חני", "f"), ("אהרון", "m"), ("בתיה", "f"), ("זיו", "m"),
    ("נעמה", "f"), ("חיים", "m"), ("יפעת", "f"), ("אליהו", "m"), ("דבורה", "f"), ("יוסי", "m"), ("צביה", "f"),
    ("שמעון", "m"), ("אורנה", "f"), ("מתן", "m"), ("עדי", "f"), ("נתנאל", "m"), ("הגר", "f"), ("אביתר", "m"),
    ("מירב", "f"), ("ברק", "m"), ("סתיו", "f"),
]

ITEMS_EN = [
    "badge", "laptop", "projector remote", "van keys", "ledger", "camera", "toolbox", "tablet", "server key",
    "access card", "headset", "tripod", "passport folder", "signed contract", "USB drive", "flashlight",
    "radio", "first-aid kit", "parking pass", "blueprint", "stamp", "microphone", "charger", "notebook",
    "safe key", "gift box", "invoice binder", "sample kit", "drill", "ladder", "coffee machine", "whiteboard",
    "hard drive", "scanner", "megaphone", "umbrella", "map", "stopwatch", "thermometer", "keycard",
]

ITEMS_HE = [
    "התג", "המחשב הנייד", "השלט של המקרן", "המפתחות של הטנדר", "הפנקס", "המצלמה", "ארגז הכלים", "הטאבלט",
    "המפתח של השרת", "כרטיס הכניסה", "האוזניות", "החצובה", "תיקיית הדרכונים", "החוזה החתום", "הדיסק און קי",
    "הפנס", "מכשיר הקשר", "ערכת העזרה הראשונה", "תו החניה", "השרטוט", "החותמת", "המיקרופון", "המטען",
    "המחברת", "המפתח של הכספת", "קופסת המתנה", "קלסר החשבוניות", "ערכת הדוגמאות", "המקדחה", "הסולם",
    "מכונת הקפה", "הלוח המחיק", "הדיסק הקשיח", "הסורק", "המגאפון", "המטריה", "המפה", "שעון העצר",
    "המדחום", "הכרטיס המגנטי",
]

PLACES_EN = [
    "storage room", "front desk", "meeting room", "server room", "loading dock", "kitchen", "archive", "lab",
    "parking garage", "lobby", "workshop", "mail room", "break room", "print room", "security office",
    "warehouse", "reception", "roof deck", "basement", "library",
]

# (definite, in, to, from)
PLACES_HE = [
    ("המחסן", "במחסן", "למחסן", "מהמחסן"), ("הקבלה", "בקבלה", "לקבלה", "מהקבלה"),
    ("חדר הישיבות", "בחדר הישיבות", "לחדר הישיבות", "מחדר הישיבות"),
    ("חדר השרתים", "בחדר השרתים", "לחדר השרתים", "מחדר השרתים"),
    ("רציף הפריקה", "ברציף הפריקה", "לרציף הפריקה", "מרציף הפריקה"), ("המטבח", "במטבח", "למטבח", "מהמטבח"),
    ("הארכיון", "בארכיון", "לארכיון", "מהארכיון"), ("המעבדה", "במעבדה", "למעבדה", "מהמעבדה"),
    ("החניון", "בחניון", "לחניון", "מהחניון"), ("הלובי", "בלובי", "ללובי", "מהלובי"),
    ("הסדנה", "בסדנה", "לסדנה", "מהסדנה"), ("חדר הדואר", "בחדר הדואר", "לחדר הדואר", "מחדר הדואר"),
    ("פינת הקפה", "בפינת הקפה", "לפינת הקפה", "מפינת הקפה"),
    ("חדר ההדפסה", "בחדר ההדפסה", "לחדר ההדפסה", "מחדר ההדפסה"),
    ("חדר האבטחה", "בחדר האבטחה", "לחדר האבטחה", "מחדר האבטחה"),
    ("המחסן הגדול", "במחסן הגדול", "למחסן הגדול", "מהמחסן הגדול"), ("העמדה", "בעמדה", "לעמדה", "מהעמדה"),
    ("המרפסת", "במרפסת", "למרפסת", "מהמרפסת"), ("המרתף", "במרתף", "למרתף", "מהמרתף"), ("הספרייה", "בספרייה", "לספרייה", "מהספרייה"),
]
