import json

COUNTRY_MAP = {
    8: "Albania",
    20: "Andorra",
    31: "Azerbaijan",
    32: "Argentina",
    36: "Australia",
    40: "Austria",
    50: "Bangladesh",
    51: "Armenia",
    68: "Bolivia",
    70: "Bosnia Herzegovina",
    76: "Brazil",
    100: "Bulgaria",
    104: "Myanmar",
    112: "Belarus",
    124: "Canada",
    152: "Chile",
    156: "China",
    158: "Taiwan ROC",
    170: "Colombia",
    191: "Croatia",
    196: "Cyprus",
    203: "Czechia",
    208: "Denmark",
    218: "Ecuador",
    233: "Estonia",
    231: "Ethiopia",
    246: "Finland",
    250: "France",
    268: "Georgia",
    276: "Germany",
    300: "Greece",
    320: "Guatemala",
    344: "Hong Kong SAR",
    348: "Hungary",
    352: "Iceland",
    356: "India",
    360: "Indonesia",
    364: "Iran",
    368: "Iraq",
    380: "Italy",
    392: "Japan",
    398: "Kazakhstan",
    400: "Jordan",
    404: "Kenya",
    410: "South Korea",
    417: "Kyrgyzstan",
    422: "Lebanon",
    434: "Libya",
    440: "Lithuania",
    446: "Macao SAR",
    458: "Malaysia",
    462: "Maldives",
    484: "Mexico",
    496: "Mongolia",
    499: "Montenegro",
    504: "Morocco",
    528: "Netherlands",
    554: "New Zealand",
    558: "Nicaragua",
    566: "Nigeria",
    578: "Norway",
    586: "Pakistan",
    604: "Peru",
    608: "Philippines",
    616: "Poland",
    620: "Portugal",
    630: "Puerto Rico",
    642: "Romania",
    643: "Russia",
    688: "Serbia",
    702: "Singapore",
    703: "Slovakia",
    705: "Slovenia",
    724: "Spain",
    752: "Sweden",
    756: "Switzerland",
    762: "Tajikistan",
    764: "Thailand",
    788: "Tunisia",
    792: "Turkey",
    804: "Ukraine",
    807: "North Macedonia",
    818: "Egypt",
    826: "Great Britain",
    840: "United States",
    858: "Uruguay",
    860: "Uzbekistan",
    862: "Venezuela",
    704: "Vietnam",
    716: "Zimbabwe",
}
SETTLEMENT_SIZE_MAP = {
    1: "under 2,000",
    2: "2,000-5,000",
    3: "5,000-10,000",
    4: "10,000-20,000",
    5: "20,000-50,000",
    6: "50,000-100,000",
    7: "100,000-500,000",
    8: "500,000 and more",
}
SETTLEMENT_TYPE_MAP = {
    1: "capital city",
    2: "regional center",
    3: "district center",
    4: "city/town",
    5: "village",
}
URBAN_RURAL_MAP = {1: "urban", 2: "rural"}
SEX_MAP = {1: "male", 2: "female"}
CITIZEN_MAP = {
    1: "a citizen of this country",
    2: "not a citizen of this country",
}
LIVE_WITH_PARENTS_MAP = {
    1: "do not live with parents",
    2: "live with your parent(s)",
    3: "live with your parent(s)-in-law",
    4: "live with both your parent(s) and parent(s)-in-law",
}
LANGUAGE_MAP = {
    30: "Afar",
    40: "Afrikaans",
    100: "Albanian",
    140: "Amharic",
    170: "Arabic",
    200: "Armenian; Hayeren",
    230: "Assyrian Neo-Aramaic",
    245: "Auslan",
    250: "Avar; Avaric",
    290: "Aymara",
    310: "Azerbaijani; Azeri",
    350: "Balinese",
    370: "Balochi",
    410: "Banjar",
    460: "Batak",
    490: "Bengali; Bangla",
    500: "Berber; Amazigh; Tamaziɣt",
    520: "Betawi",
    550: "Bikol; Bicolano",
    610: "Romblomanon",
    630: "Blaan",
    680: "Brahui",
    710: "Buginese/Bugis",
    720: "Bulgarian",
    740: "Burmese",
    790: "Cantonese",
    810: "Catalan; Valencian",
    820: "Cebuano; Bisaya; Binisaya",
    850: "Chavacano; Chabacano",
    860: "Chechen",
    890: "Karanga; Korekore",
    910: "Ndau; chiNdau",
    950: "Chitoko",
    1030: "Croatian",
    1040: "Czech",
    1100: "Danish",
    1240: "English",
    1260: "Esan",
    1270: "Spanish; Castilian",
    1290: "Estonian",
    1360: "Filipino; Pilipino",
    1400: "French",
    1490: "Garifuna",
    1530: "German",
    1540: "Gilaki",
    1580: "Greek, Modern",
    1600: "Guarani",
    1610: "Gujarati",
    1670: "Hakka Chinese",
    1695: "Hassaniyya, Klem El Bithan",
    1700: "Hausa",
    1730: "Hiligaynon; Ilonggo",
    1740: "Hindi",
    1770: "Hungarian",
    1850: "Igbo",
    1880: "Ilo Ilocano; Ilokano; Iloko",
    1890: "Indonesian",
    1930: "Pamiri languages",
    1980: "Isoko",
    1990: "Italian",
    2000: "Itneg",
    2020: "Japanese",
    2030: "Javanese",
    2100: "Kalanga",
    2103: "Kalenjin",
    2120: "Kamayo",
    2126: "Kamba",
    2170: "Kapampangan",
    2180: "Kaqchikel",
    2210: "Kashmiri",
    2220: "Sgaw Karen; Sgaw Kayin; Karen",
    2230: "Kazakh",
    2270: "Central Khmer",
    2280: "Kikuyu; Gikuyu",
    2310: "Kirghiz; Kyrgyz",
    2316: "Kisii",
    2390: "Korean",
    2420: "Kurdish; Yezidi",
    2480: "Lampung",
    2500: "Lao",
    2530: "Mayan languages",
    2560: "Lezgian; Lezgi; Lezgin",
    2657: "Luhya",
    2670: "Lurish; Luri; Bakhtiari",
    2720: "Luo, Lwo; Lwoian",
    2740: "Madurese",
    2760: "Maguindanao",
    2790: "Makassarese",
    2810: "Malay; Malaysian",
    2820: "Malayalam",
    2840: "Maltese",
    2870: "Standard Chinese; Mandarin; Putonghua; Guoyu",
    2920: "Maori",
    2930: "Maranao",
    2940: "Marathi",
    2969: "Maasai",
    2981: "Meru",
    2987: "Mijikenda",
    3020: "Mon",
    3030: "Mongolian",
    3100: "Muong",
    3200: "North Ndebele",
    3234: "Northern Thai; Lanna",
    3390: "Oromo",
    3420: "Palembang",
    3490: "Persian; Farsi; Dari",
    3510: "Nigerian Pidgin",
    3520: "Polish",
    3530: "Portuguese",
    3540: "Punjabi, Panjabi",
    3550: "Pashto, Pushto",
    3570: "Quechua",
    3580: "Romanian, Moldavian, Moldovan",
    3600: "Romansh",
    3610: "Romani; Romany",
    3630: "Russian",
    3670: "Sama-Bajaw",
    3720: "Saraiki",
    3780: "Serbian",
    3810: "Shan",
    3830: "Shona; chiShona",
    3840: "Sidamo; Sidaama; Sidaamu Afoo",
    3860: "Sindhi",
    3870: "Sinhala, Sinhalese",
    3890: "Slovak",
    3920: "Somali",
    3992: "Southern Thai; Dambro; Pak Thai",
    4040: "Sundanese",
    4060: "Surigaonon",
    4075: "Swahili",
    4110: "Swedish",
    4130: "Tagalog",
    4150: "Hokkien; Minnan",
    4160: "Tajik",
    4190: "Tamil",
    4200: "Tatar",
    4210: "Tausug",
    4220: "Telugu",
    4230: "Thai; Central Thai",
    4260: "Tigrinya",
    4280: "Tiv",
    4295: "Tonga",
    4310: "Toraja-Saʼdan",
    4360: "Tunisian Arabic; Tunisian",
    4365: "Turkana",
    4370: "Turkish",
    4380: "Turkmen",
    4400: "Uighur, Uyghur",
    4410: "Ukrainian",
    4420: "Urdu",
    4430: "Urhobo",
    4450: "Uzbek",
    4460: "ven Venda; Tshivenda",
    4470: "Vietnamese",
    4520: "Waray",
    4580: "Yakan",
    4610: "Yiddish",
    4620: "Yoruba",
    9040: "Other European",
    9060: "Other Chinese dialects",
    9900: "Other local; aboriginal; tribal; community",
}
MARITAL_STATUS_MAP = {
    1: "married",
    2: "living together as married",
    3: "divorced",
    4: "separated",
    5: "widowed",
    6: "single",
}
EDUCATION_MAP = {
    0: "no formal education",
    1: "primary education",
    2: "lower secondary education",
    3: "upper secondary education",
    4: "post-secondary non-tertiary education",
    5: "short-cycle tertiary education",
    6: "Bachelor's degree",
    7: "Master's degree",
    8: "Doctoral degree",
}
EMPLOYMENT_STATUS_MAP = {
    1: "employed full-time",
    2: "employed part-time",
    3: "self-employed",
    4: "retired/pensioned",
    5: "a housewife",
    6: "a student",
    7: "unemployed",
}
OCCUPATION_MAP = {
    0: "never had a job",
    1: "work in professional/technical fields",
    2: "work in higher administrative positions",
    3: "work in clerical jobs",
    4: "work in sales",
    5: "work in service",
    6: "work as a skilled worker",
    7: "work as a semi-skilled worker",
    8: "work as an unskilled worker",
    9: "work as a farm worker",
    10: "are a farm proprietor/manager",
}
OCCUPATION_MAP_THIRD = {
    0: "never had a job",
    1: "works in professional/technical fields",
    2: "works in higher administrative positions",
    3: "works in clerical jobs",
    4: "works in sales",
    5: "works in service",
    6: "works as a skilled worker",
    7: "works as a semi-skilled worker",
    8: "works as an unskilled worker",
    9: "works as a farm worker",
    10: "is a farm proprietor/manager",
}
OCCUPATION_MAP_PAST = {
    0: "never had a job",
    1: "worked in professional/technical fields",
    2: "worked in higher administrative positions",
    3: "worked in clerical jobs",
    4: "worked in sales",
    5: "worked in service",
    6: "worked as a skilled worker",
    7: "worked as a semi-skilled worker",
    8: "worked as an unskilled worker",
    9: "worked as a farm worker",
    10: "was a farm proprietor/manager",
}
SECTOR_EMPLOYMENT_MAP = {
    1: "government/public sector",
    2: "private business/industry",
    3: "private non-profit sector",
}
CHIEF_WAGE_EARNER_MAP = {
    1: "are the chief wage earner",
    2: "are not the chief wage earner",
}
FAMILY_SAVINGS_MAP = {
    1: "had saved money",
    2: "just got by",
    3: "had spent some savings",
    4: "had spent savings and borrowed money",
}
SOCIAL_CLASS_MAP = {
    1: "upper class",
    2: "upper middle class",
    3: "lower middle class",
    4: "working class",
    5: "lower class",
}
INCOME_SCALE_MAP = {i: f"group {i}" for i in range(1, 11)}
RELIGION_MAJOR_MAP = {
    0: "no religion",
    1: "Roman Catholic",
    2: "Protestant",
    3: "Orthodox",
    4: "Jewish",
    5: "Muslim",
    6: "Hindu",
    7: "Buddhist",
    8: "Other",
}
ETHNIC_GROUP_MAP = {
    20001: "Caucasian white",
    20002: "Negro black",
    20003: "South Asian (Indian, Pakistani..)",
    20004: "East Asian (Chinese, Japanese...)",
    20005: "Arabic, Central Asian",
    32001: "White",
    32002: "Light brown",
    32003: "Dark brown",
    32004: "Black",
    32005: "Indigenous",
    36001: "Australian (English speaking)",
    36002: "European",
    36003: "South Asian (Indian, Pakistani, etc)",
    36004: "East Asian (Chinese, Japanese, etc)",
    36005: "Arabic, Central Asian",
    36006: "Southeast Asian: Thai, Vietnamese, Malaysian, etc",
    36007: "Aboriginal or Torres Strait Islander",
    50006: "Bengali",
    50007: "Chakma",
    50008: "Murong",
    51001: "Armenian",
    51005: "Russian",
    51006: "Yazidis",
    68001: "Not pertaining to Indigenous groups",
    68002: "Quechua",
    68003: "Aymara",
    68004: "Guaraní",
    68005: "Chiquitano",
    68006: "Mojeño",
    68007: "Afroboliviano",
    68008: "Indigenous with no further detail",
    76001: "White",
    76002: "Black",
    76003: "Brown - Moreno ou pardo",
    76004: "Oriental: Chines, Japanese,...",
    76005: "Indigenous",
    104001: "Bamar",
    104002: "Kayin",
    104003: "Rakhine",
    104004: "Shan",
    104005: "Mon",
    124001: "Caucasian (White)",
    124002: "Black (African, African-American, etc.)",
    124003: "West Asian (Iranian, Afghan, etc.)",
    124004: "Southeast Asian (Vietnamese, Cambodian, Malaysian, etc.)",
    124005: "Arabic (Central Asia)",
    124006: "South Asian (Indian, Bangladeshi, Pakistani, Sri Lankan, etc.)",
    124007: "Latin American / Hispanic",
    124008: "Aboriginal / First Nations",
    124009: "Chinese",
    124010: "Filipino",
    124011: "Korean",
    124012: "Japanese",
    152001: "White, Caucasian",
    152002: "Black",
    152008: "Indigenous",
    152009: "Asiatic",
    152012: "Mestizo(a)",
    152013: "Mulatto(a)",
    156001: "Chinese",
    158001: "Hakka from Taiwan",
    158002: "Minnanese from Taiwan",
    158003: "Mainlander/China",
    158004: "Aboriginal",
    170008: "Afro-colombian",
    170009: "Gypsie",
    170010: "Indigenous",
    170011: "White",
    196001: "Caucasian white",
    196003: "South Asian Indian, Pakistani, etc.",
    196005: "Arabic, Central Asian",
    203004: "Slovak",
    203005: "Poland",
    203006: "Ukrainian",
    218011: "Blanco",
    218012: "Mestizo",
    218013: "Negro",
    218014: "Indígena",
    218017: "Montubio",
    218018: "Mulato",
    231001: "Amhara",
    231002: "Tigre",
    231003: "Oromo",
    231004: "Somali",
    231005: "Afar",
    231006: "Sidama",
    231007: "Wolayta",
    231998: "Other Africans/Negro Black",
    300001: "Caucasian white",
    300002: "Negro Black",
    300004: "East Asian Chinese, Japanese, etc.",
    300005: "Arabic, Central Asian",
    320001: "Ladino",
    320002: "Cross breed",
    320003: "Brown",
    320004: "Indigenous",
    320005: "White",
    320006: "Hispanic",
    344001: "Chinese",
    344002: "Filipino",
    344003: "Indonesian",
    344004: "White",
    344005: "Indian",
    344006: "Nepalese",
    344007: "Pakistani",
    344008: "Thai",
    344998: "Other Asian",
    356017: "Hindu (Scheduled Castes)",
    356018: "Hindu (Scheduled Tribes)",
    356019: "Hindu (Other Backward Castes)",
    356024: "General",
    360002: "Chinese",
    360004: "Javanese",
    360005: "Sundanese",
    360007: "Aceh",
    360008: "Batak",
    360009: "Banjar",
    360010: "Betawi",
    360011: "Bengkulu",
    360012: "Bugis",
    360013: "Dani",
    360014: "Dayak",
    360015: "Flores",
    360017: "Lampung",
    360018: "Maduranese",
    360019: "Makassar",
    360020: "Mandar",
    360021: "Manggarai",
    360022: "Melayu",
    360023: "Minangkabau",
    360024: "Palembang",
    360026: "Toraja",
    364001: "Persian",
    364002: "Turk/Azeri",
    364003: "Kurd",
    364004: "Lor",
    364005: "Gilak/Mazani/Shomali",
    364006: "Baluch",
    364007: "Arab",
    368001: "Arab",
    368002: "Kurdish",
    368003: "Turk",
    368004: "Ashur",
    398001: "Korean",
    398002: "Uigur",
    398003: "Bashkir",
    398004: "Lezgin",
    398005: "Belorus",
    398021: "Azeri",
    398022: "Iranian and Central Asian",
    398023: "Georgian",
    398024: "German",
    398025: "Kazakh",
    398026: "Kurdish",
    398027: "Kyrgyz",
    398028: "Moldovan",
    398029: "Russian",
    398030: "Tajik",
    398031: "Tatar",
    398032: "Ukrainian",
    398033: "Uzbek",
    398040: "Turkish",
    404001: "Kalenjin",
    404002: "Kamba",
    404003: "Kikuyu",
    404004: "Kisii",
    404005: "Luhya",
    404006: "Luo",
    404007: "Maasai",
    404008: "Meru",
    404009: "Mijikenda",
    404010: "Somalis",
    404011: "Turkana",
    404014: "Arabs in Kenya (Arabic, Central Asian)",
    410004: "East Asian (Chinese, Japanese, etc)",
    417001: "Kirguís",
    417002: "European",
    417003: "Tayiko",
    417004: "Ruso",
    417005: "Kazakh",
    434005: "Arabic",
    434006: "Amazigh",
    434007: "Tuareg",
    434008: "Toubou",
    446001: "Caucasian white",
    446003: "South Asian Indian, Pakistani, etc.",
    446004: "East Asian Chinese, Japanese, etc.",
    446006: "Chinese",
    446007: "Portuguese/Macaense",
    446008: "Southeast Asians (Indonesia, Philippines, Thailand)",
    458004: "Malay",
    458012: "Chinese",
    458014: "Indian",
    462001: "Caucasian white",
    462003: "South Asian India, Pakistani, etc",
    462004: "East Asian Chinese, Japanese, etc",
    484001: "White",
    484002: "Light brown",
    484003: "Dark brown",
    484004: "Black",
    484005: "Indigenous",
    496001: "Khalkh",
    496002: "Dorvod",
    496003: "Bayad",
    496004: "Buriad",
    496005: "Zakhchin",
    496006: "Myangad",
    496007: "Uuld",
    496008: "Kazakh",
    496009: "Tuva",
    496011: "Dariganga",
    496012: "Uzemchin",
    496014: "Uriankhai",
    496015: "Khoton",
    496016: "Darkhad",
    496017: "Torguud",
    504005: "Arabe",
    528001: "Caucasian white",
    528002: "Negro Black",
    528003: "South Asian Indian, Pakistani, etc.",
    528004: "East Asian Chinese, Japanese, etc.",
    528005: "Arabic, Central Asian",
    558001: "Half Blood",
    558002: "Indigenous",
    558003: "Afrocaribeño",
    566001: "Yoruba",
    566002: "Hausa",
    566003: "Igbo",
    566004: "Fulani",
    566005: "Tiv",
    566006: "Ibibio",
    566998: "Other Africans",
    586001: "Punjabi",
    586003: "Baluchi",
    586004: "Sindhi",
    586005: "Urdu speaking",
    586006: "Pashto",
    586007: "Hindko",
    586008: "Seraiki",
    586009: "Hindko",
    604001: "White",
    604012: "Indigenous / Native",
    604013: "Indigenous half-breed",
    604014: "Afro half-breed",
    604015: "Asian half-breed",
    604016: "European half-breed",
    604998: "Migrant of other origin",
    608001: "Tagalog",
    608002: "Bisaya",
    608003: "Ilonggo",
    608004: "Bicolano",
    608005: "Ilocano",
    608006: "Waray",
    608007: "Chabacano",
    608010: "Kapampangan",
    608015: "Aklanon",
    608017: "Sama",
    608018: "Matanao",
    608020: "Bilaan",
    608024: "Cebuano",
    608026: "Antiqueno",
    608027: "Masbateno",
    608028: "Pangasinense",
    608031: "Tausog",
    608032: "Suriganon",
    608039: "Maguindanao",
    608047: "Igorot",
    608048: "Yakan",
    608049: "Marinduque",
    608050: "Ayangan (Kankanaey)",
    608051: "Tinguian Tribe",
    608052: "Belwang tribe",
    608053: "Matinguian Tribe",
    608054: "Sambal",
    608055: "Mangyan",
    608056: "Romblomanon",
    608057: "Subanin",
    608058: "Cantilangnon",
    608059: "Kamayo",
    608060: "Boholano",
    608061: "Taga Kaulo",
    608062: "Sinamah",
    630001: "White",
    630002: "Black",
    630007: "Indigenous",
    630012: "Light brown",
    630013: "Dark brown",
    642005: "Caucasian white",
    643001: "Russian",
    643002: "Tatar",
    643003: "Ukrainian",
    643005: "Jew",
    643007: "German",
    643015: "Georgian",
    643016: "Armenian",
    643020: "Moldovan",
    643023: "Italian",
    643035: "Kazakh",
    643036: "Azeri",
    643037: "North-East Asian",
    643040: "Tajik",
    643053: "Kyrgyz",
    643080: "Uzbek",
    643097: "Iranian and Central Asian",
    688001: "Caucasian white",
    702001: "Caucasian white",
    702003: "South Asian Indian, Pakistan, etc.",
    702004: "Chinese",
    702006: "Malay",
    702008: "Eurasian",
    703001: "Gypsy",
    703005: "Slovak",
    703006: "Hungarian",
    703008: "Ruthenian",
    703009: "Bohemia/Czech Republic",
    704001: "Kinh",
    704002: "Muong",
    704007: "Thai",
    704009: "China",
    716001: "Africans/Negro Black",
    716002: "Caucasian White",
    716006: "Shona",
    716007: "Ndebele",
    716008: "Arabic, Central Asian",
    762001: "Tajik",
    762002: "Uzbek",
    762004: "Russian",
    764001: "Thai",
    764002: "China",
    764003: "Malayu",
    764004: "Tribe",
    788002: "Negro Black",
    788003: "Tamazight (Berber)",
    788005: "Arabic",
    804001: "Ukrainians",
    804002: "Russians",
    804003: "Belarusians",
    804004: "Tatars",
    804005: "Jews",
    818001: "Arab",
    818004: "Noba",
    818006: "Coptic",
    826002: "Black-Caribbean",
    826003: "Black-African",
    826004: "Black-Other",
    826005: "Indian",
    826006: "Pakistani",
    826007: "Bangladeshi",
    826008: "Chinese",
    826015: "Arabic, Central Asian",
    826016: "Mixed race",
    826100: "English / Welsh / Scottish / Northern Irish / British",
    826101: "Irish",
    826102: "Gypsy or Irish Traveller",
    826103: "other White background",
    826104: "White and Black Caribbean",
    826105: "White and Black African",
    826106: "White and Asian",
    826107: "other Asian background",
    840001: "White, non-Hispanic",
    840002: "Black, Non-Hispanic",
    840003: "Other, Non-Hispanic",
    840004: "Hispanic",
    840005: "Two plus, non-Hispanic",
    858100: "White",
    858101: "Afro or black",
    858102: "Indigenous",
    858103: "Half Blood",
    858104: "Mulatto",
    858105: "Asian",
    860001: "Uzbek",
    860002: "Russian",
    860003: "Tatarin",
    860004: "Kazakhs",
    860005: "Karakalpak",
    860006: "Tajik",
    860007: "Kyrgyz",
    860008: "Turkmen",
    862001: "White",
    862002: "Mestizo / Light brown",
    862003: "Dark-skinned / Dark-brown",
    862004: "Black",
    862005: "Indigenous",
    909002: "English / Welsh / Scottish / Northern Irish / British",
    909003: "Irish",
    909005: "other White background",
    909007: "White and Black African",
    909010: "Indian",
    909013: "Chinese",
    909014: "other Asian background",
    909015: "African",
    909016: "Caribbean",
    909017: "Black / African / Caribbean background",
}


def describe_respondent(respondent, extended=False):
    parts = []

    sex = SEX_MAP.get(respondent.get("sex"))
    if sex:
        parts.append(f"You are a {sex}")

    age = respondent.get("age")
    if age is not None and int(age) >= 0:
        parts.append(f"You are aged {age}")

    country = COUNTRY_MAP.get(respondent.get("country"))
    if country:
        parts.append(f"You live in {country}")
    citizen = CITIZEN_MAP.get(respondent.get("citizen"))
    if citizen:
        parts.append(f"You are {citizen}")

    # birth_country = COUNTRY_MAP.get(respondent.get("birth_country"))
    # birth_country_mother = COUNTRY_MAP.get(respondent.get("birth_country_mother"))
    # birth_country_father = COUNTRY_MAP.get(respondent.get("birth_country_father"))
    # if birth_country:
    #     parts.append(f"You were born in {birth_country}")
    # if birth_country_mother:
    #     parts.append(f"Your mother was born in {birth_country_mother}")
    # if birth_country_father:
    #     parts.append(f"Your father was born in {birth_country_father}")

    immigrant_status = respondent.get("immigrant_self")
    if immigrant_status is not None and int(immigrant_status) >= 0:
        if immigrant_status == 1:
            parts.append("You are born locally")
        else:
            parts.append("You are an immigrant")
    immigrant_status_mother = respondent.get("immigrant_mother")
    if immigrant_status_mother is not None and int(immigrant_status_mother) >= 0:
        if immigrant_status_mother == 1:
            parts.append("Your mother is born locally")
        else:
            parts.append("Your mother is an immigrant")
    immigrant_status_father = respondent.get("immigrant_father")
    if immigrant_status_father is not None and int(immigrant_status_father) >= 0:
        if immigrant_status_father == 1:
            parts.append("Your father is born locally")
        else:
            parts.append("Your father is an immigrant")

    urban_rural = URBAN_RURAL_MAP.get(respondent.get("urban_rural"))
    settlement_type = SETTLEMENT_TYPE_MAP.get(respondent.get("settlement_type"))
    settlement_size = SETTLEMENT_SIZE_MAP.get(respondent.get("settlement_size"))
    settlement_parts = []
    if urban_rural:
        settlement_parts.append(f"{urban_rural} area")
    if settlement_type:
        settlement_parts.append(f"{settlement_type}")
    if settlement_size:
        settlement_parts.append(f"with a population of {settlement_size}")
    if settlement_parts:
        parts.append(f"You live in a {', '.join(settlement_parts)}")

    household_size = respondent.get("household_size")
    live_with_parents = LIVE_WITH_PARENTS_MAP.get(respondent.get("live_with_parents"))
    language_home = LANGUAGE_MAP.get(respondent.get("home_language"))
    if household_size is not None and int(household_size) >= 0:
        parts.append(f"Your household consists of {household_size} members")
    if live_with_parents:
        parts.append(f"You {live_with_parents}")
    if language_home:
        parts.append(f"You speak {language_home} at home")

    marital_status = MARITAL_STATUS_MAP.get(respondent.get("marital_status"))
    num_children = respondent.get("have_children")
    marital_parts = []
    if marital_status:
        marital_parts.append(f"are {marital_status}")
    if num_children is not None and int(num_children) >= 0:
        marital_parts.append(f"have {num_children} children")
    if marital_parts:
        parts.append(f"You {', '.join(marital_parts)}")

    education_self = EDUCATION_MAP.get(respondent.get("education_self"))
    education_spouse = EDUCATION_MAP.get(respondent.get("education_spouse"))
    education_father = EDUCATION_MAP.get(respondent.get("education_father"))
    education_mother = EDUCATION_MAP.get(respondent.get("education_mother"))
    if education_self:
        parts.append(f"Your education level is {education_self}")
    if education_spouse:
        parts.append(f"Your spouse's education level is {education_spouse}")
    if education_father:
        parts.append(f"Your father's education level is {education_father}")
    if education_mother:
        parts.append(f"Your mother's education level is {education_mother}")

    employment_self = EMPLOYMENT_STATUS_MAP.get(respondent.get("employment_self"))
    occupation_self = OCCUPATION_MAP.get(respondent.get("occupation_self"))
    if employment_self:
        parts.append(f"You are {employment_self}")
    if occupation_self:
        parts.append(f"You {occupation_self}")
    sector_employment = SECTOR_EMPLOYMENT_MAP.get(respondent.get("work_sector"))
    if sector_employment:
        parts.append(f"You work in the {sector_employment}")

    employment_spouse = EMPLOYMENT_STATUS_MAP.get(respondent.get("employment_spouse"))
    occupation_spouse = OCCUPATION_MAP_THIRD.get(respondent.get("occupation_spouse"))
    if employment_spouse:
        parts.append(f"Your spouse is {employment_spouse}")
    if occupation_spouse:
        parts.append(f"Your spouse {occupation_spouse}")

    occupation_father14 = OCCUPATION_MAP_PAST.get(respondent.get("occupation_father14"))
    if occupation_father14:
        parts.append(f"Your father {occupation_father14} when you were 14 years old")

    chief_wage_earner = CHIEF_WAGE_EARNER_MAP.get(respondent.get("chief_wage_earner"))
    family_savings = FAMILY_SAVINGS_MAP.get(respondent.get("saving_behavior"))
    social_class = SOCIAL_CLASS_MAP.get(respondent.get("social_class"))
    if chief_wage_earner:
        parts.append(f"You {chief_wage_earner}")
    if family_savings:
        parts.append(f"During the past year, your family {family_savings}")
    if social_class:
        parts.append(f"You belong to the {social_class}")

    income_scale = INCOME_SCALE_MAP.get(respondent.get("income_decile"))
    if income_scale:
        parts.append(
            f"Your income is in income group {income_scale} (1 = lowest, 10 = highest)"
        )

    religion = RELIGION_MAJOR_MAP.get(respondent.get("religion"))
    if religion:
        if religion == "no religion":
            parts.append("You have no religious affiliation")
        elif religion == "Other":
            parts.append("You identify with another religion")
        else:
            parts.append(f"You identify as {religion}")

    ethnic_group = ETHNIC_GROUP_MAP.get(respondent.get("ethnic_group"))
    if ethnic_group:
        parts.append(f"Your ethnic group is {ethnic_group}")

    return ". ".join(parts) + "."


if __name__ == "__main__":
    with open("demographic/wvs_demographic_500.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    descriptions = {}
    for i, respondent in enumerate(data, start=1):
        descriptions[i] = describe_respondent(respondent)

    with open(
        "descriptions/wvs_demographic_descriptions_500.json", "w", encoding="utf-8"
    ) as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)

    print("Saved descriptions to json file")
