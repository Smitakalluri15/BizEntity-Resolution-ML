# Business Address Noise Catalog

This catalog documents the empirical address noise patterns discovered by comparing canonical Source 1 records to their true matching entities in Source 2 and Source 3.

---

## 1. Street and Secondary Unit Abbreviations

Standard postal abbreviations are used interchangeably with full forms, contractions, and punctuated variations.

### Observed Abbreviation Pairs:
- **`Street` vs `St` vs `ST`:**
  - **S1:** `"2756 Greeley Street, Schenectady, NY"`
  - **S2 (`S2-414630023`):** `"2756 GREELEY ST, SCHENECTADY, NY"`
  - **S1:** `"132 Pulaski Street, Brooklyn, NY"`
  - **S2 (`S2-625512512`):** `"332 PULASKI ST, BROOKLYN, NY"`
- **`Avenue` vs `Ave` vs `AVE`:**
  - **S1:** `"243 Mountain Avenue, Arlington, MA"`
  - **S2 (`S2-112921551`):** `"243 MOUNTAIN AVE, ARLINGTON CDP, MA"`
  - **S1:** `"1836 Robin Avenue, MT, Billings, Bldg Lot 9"`
  - **S3 (`S3-374962673`):** `"1836 Robin Ave, Bldg Lot 9, Billings, Montana"`
- **`Road` vs `Rd` vs `RD`:**
  - **S1:** `"861 River Road, Windham, ME"`
  - **S2 (`S2-440166852`):** `"861 RIVER RD, WINDHAM, ME"`
  - **S1:** `"6457 Chief Washakie Road, Casper, WY"`
  - **S2 (`S2-53015605`):** `"6457-B CHIEF WASHAKIE RD, CASPER, WY"`
- **`Drive` vs `Dr`:**
  - **S1:** `"1129 Cherry Ridge Drive, Sugarcreek, OH"`
  - **S3 (`S3-319489488`):** `"1129 Cherry Ridge Dr, NULL, Sugarcreek, Ohio"`
- **`Court` vs `Ct`:**
  - **S1:** `"2488 Pierce Court, Simi Valley, CA"`
  - **S3 (`S3-727461540`):** `"2488 Pierce Ct, City Of Simi Valley, California"`
- **`Lane` vs `Ln`:**
  - **S1:** `"10866 Verbena Lane, Scottsdale, Fl 1, AZ"`
  - **S2 (`S2-62160982`):** `"10866 VERBENA LN, SCOTTSADLE, AZ"`
  - **S1:** `"3068 Crystal Ridge Lane, Colton, CA"`
  - **S2 (`S2-404216086`):** `"COLTON TOWNSHIIP, CA, 3068 Crystal Ridge Ln"`
- **`Floor` vs `Fl` vs `1St / 3Rd Floor`:**
  - **S1:** `"301-304, 3Rd Floor, Campus 31, Rmz Ecoworld, Sarjapur Marathahalli Orr, Bangalore, Karnataka"`
  - **S3 (`S3-752044877`):** `"3Rd Floor, Campus 31, Rmz Ecoworld, Sarjapur Marathahalli Orr, KA, Bangalore, 301-304"`
- **`Apartment` / `Suite` / `Unit` (`Apt`, `STE`, `Unit No.`):**
  - **S1:** `"Unit No. 202 & 203, Sneh Signature, Tulsidham Gidc Road, Vadodara, Gujarat"`
  - **S2 (`S2-78586487`):** `"UNIT NO. 202 & 203, SNEH SIGNATURE, TULSIDHAM GIDC ROAD, VADODARA, Gujarat"`
  - **S3 (`S3-728592441`):** `"369 Vine St, Unit STE C, Harrisonburg, Virginia"`

---

## 2. State Names: Abbreviations vs Full Names

Records oscillate between 2-letter state postal codes and full state names (in US and India).

### Examples:
- **US States (`CA` vs `California`, `OH` vs `Ohio`, `MN` vs `Minnesota`, `IL` vs `Illinois`, `WY` vs `Wyoming`):**
  - **S1:** `"1129 Cherry Ridge Drive, Sugarcreek, OH"`
  - **S3 (`S3-319489488`):** `"1129 Cherry Ridge Dr, NULL, Sugarcreek, Ohio"`
  - **S1:** `"2488 Pierce Court, Simi Valley, CA"`
  - **S3 (`S3-727461540`):** `"2488 Pierce Ct, City Of Simi Valley, California"`
  - **S1:** `"16461 Briarwood Drive, Effingham, IL"`
  - **S3 (`S3-23005351`):** `"6461 Briarwood Drive, Effingham, Illinois"`
- **Indian States (`TN` vs `Tamil Nadu`, `MH` vs `Maharashtra`, `KA` vs `Karnataka`, `RJ` vs `Rajasthan`, `DL` vs `Delhi`, `WB` vs `West Bengal`):**
  - **S1:** `"41, Plaza Centre, 129, G.N.Chetty Road, Chennai, Tamil Nadu"`
  - **S3 (`S3-728059936`):** `"41, Plaza Centre, 129, G.n.chetty Road, Chennai, TN"`
  - **S1:** `"F-19, Shopping Centre Mansarover Garden, New Delhi, Delhi"`
  - **S3 (`S3-35514576`):** `"Hn 817 F-19, Shopping Centre Mansarover Garden, Kirti Nagar New Delhi, DL"`
  - **S1:** `"Vill: Chaltia, Pops: Berhampore, Murshidabad, West Bengal"`
  - **S3 (`S3-73497201`):** `"Vill: Chaltia, Murshidabad, WB"`
  - **S1:** `"P. No. 17, Vijay Nagar Kartarpura Phatak Ke Pass, 22- Godam, Jaipur, Rajasthan"`
  - **S3 (`S3-286798794`):** `"P. No. 17, Vijay Nagar Kartarpura Phatak Ke Pass, 22- Godam, Jaipur, RJ"`

---

## 3. Missing Components and Truncation

Matches often suffer from severe truncation or completely missing address strings.

### Examples:
- **Completely Missing Address (`nan` / `NULL` / Empty):**
  - **S1:** `"1739 Labrador Drive, Costa Mesa, CA"`
  - **S3 (`S3-56331466`):** `nan`
  - **S1:** `"861 River Road, Windham, ME"`
  - **S3 (`S3-275800609`):** `nan`
  - *Note: 168,967 records in S2 (3.36%) and 175,916 records in S3 (3.33%) have completely null address fields!*
- **Partial Truncation (Missing House Number, City, or Street):**
  - **S1:** `"1981 Bonanza Court, Unit 102, Gilbert, AZ"`
  - **S2 (`S2-944913305`):** `"BONANZA COURT, GILBERT, AZ"` (Missing house number `1981` & `Unit 102`)
  - **S1:** `"155 Longbeach Drive, Hertford, NC"`
  - **S2 (`S2-514191558`):** `"LONGBEACH DR, HERTFORD, NC"`
  - **S1:** `"Sree Power Pvt Ltd, 39, Toovipuram West Xi Street Anna Nagar, Tuticorin, Thoothukudi, Tamil Nadu"`
  - **S3 (`S3-816916132`):** `"TN, Tuticorin, Thoothukudi, 39"` (Street name omitted entirely)
  - **S1:** `"610 Country Farms Road, Tazewell County, VA"`
  - **S3 (`S3-712837153`):** `"Country Farms Road, Tazewell County, Virginia"`

---

## 4. Landmark-Based and Contextual References

Indian addresses frequently rely on relative landmarks, surrounding infrastructure, or administrative blocks rather than standard street grids.

### Examples:
- **`Near ...` / `Behind ...` / `Opposite ...` / `Pass Ke Pass`:**
  - **S1:** `"Pl 42 New Mangalwar Peth, Nr Ladka T Pump, Pune City, Pune, Maharashtra"`
  - **S3 (`S3-530137928`):** `"Block F-496 Pl 42 New Mangalwar Peth, Nr Ladka T Pump, Pune City, Pune, महाराष्ट्र"`
  - **S1:** `"Mumbai City, Maharashtra, Mumbai, S. M. Building, 1St Floor, Office No.25/26 Behind Hinduja College, New Charni Road, 315-C"`
  - **S3 (`S3-170854488`):** `"315-C, S. M. Building, 1St Floor, Office No.25/26 Behind Hinduja College, New Charni Road, Mumbai, Mumbai City, महाराष्ट्र"`
  - **S1:** `"P. No. 17, Vijay Nagar Kartarpura Phatak Ke Pass, 22- Godam, Jaipur, Rajasthan"`
  - **S3 (`S3-286798794`):** `"P. No. 17, Vijay Nagar Kartarpura Phatak Ke Pass, 22- Godam, Jaipur, RJ"`
  - **S2 (`S2-545337877`):** `"# 34. , BEHIND PLOT NO.17(I), 2ND PHASE, 1ST C MAIN, PEENYA INDUSTRIAL AREA, ಕರ್ನಾಟಕ"`
- **Parentage / In Care Of (`C/O ...`):**
  - **S1:** `"C/O Nanhak S/O Bansu, Purwa Chetiya Bazaar, Bansi, Siddharth Nagar, Uttar Pradesh"`
  - **S2 (`S2-175745851`):** `"C/O NANHAK S/O BANSU, SIDDHARTH NAGAR, Uttar Pradesh"`

---

## 5. Address Component Re-ordering

Tokens and administrative tiers appear in arbitrary order across records.

### Examples:
- **State / City Prepended vs Appended:**
  - **S1:** `"Fowler, IN, 602 7th Street"`
  - **S2 (`S2-646039812`):** `"IN, 602- 7RD STREET, FOWLER"`
  - **S1:** `"WV, 128 River Avenue, Weston"`
  - **S3 (`S3-70790379`):** `"127 River Avenue, Westton, West Virginia"`
  - **S1:** `"1004 7th Street, Minneapolis, MN"`
  - **S3 (`S3-807314705`):** `"Minnesota, Minnepaolis, Seventh St"`
- **House Number / Plot Moved to End:**
  - **S1:** `"301-304, 3Rd Floor, Campus 31, Rmz Ecoworld, Sarjapur Marathahalli Orr, Devarabeesanahalli Village, Varthur Hobli, Bangalore South, Bangalore, Karnataka"`
  - **S3 (`S3-752044877`):** `"3Rd Floor, Campus 31, Rmz Ecoworld, Sarjapur Marathahalli Orr, Devarabeesanahalli Village, Varthur Hobli, KA, Bangalore, 301-304"`
  - **S1:** `"Mumbai, S. M. Building, Office 25/26, 315-C"`
  - **S3 (`S3-170854488`):** `"315-C, S. M. Building, Office 25/26, Mumbai"`

---

## 6. Numbering Formats, Prefixes, and Range Variations

Numerical tokens differ widely in punctuation, leading symbols, leading zeros, and range delimiters.

### Examples:
- **Hyphenated Ranges vs Single Numbers:**
  - **S1:** `"1836 Robin Avenue, MT, Billings, Bldg Lot 9"`
  - **S2 (`S2-492276871`):** `"BILLINGS, MT, 1836-1840 ROBIN AVENUE"`
  - **S1:** `"7436 Paris Avenue, Birmingham, AL"`
  - **S2 (`S2-77841735`):** `"7436-7438 PARIS AVENUE, BIRMINGHAM, AL"`
  - **S1:** `"125, Kanchan Bag, Ndore, Madhya Pradesh"`
  - **S3 (`S3-464218104`):** `"1-25, Kanchan Bag, Ndore, MP"`
- **Prefixes (`#`, `##`, `H.No`, `Door No`, `Plot -`, `P. No.`):**
  - **S1:** `"Creative Light Limited, Vaisyanazhikathu Veedu, Adichanalloor Village, Thazhuthala, Kollam, Kerala"`
  - **S2 (`S2-121982035`):** `"H.NO 24 VAISYANAZHIKATHU VEEDU, ADICHANALLOOR VILLAGE, THAZHUTHALA, KOLLAM, Kerala"`
  - **S3 (`S3-491935776`):** `"Door No 534 15, Meera, L D Ruparel Marg, Malabar Hill, Mumbai, MH"`
  - **S2 (`S2-2101517`):** `"OK, NINNEKAH CDP, ##913 Highway 277"`
- **Leading Zeros in Numerical Tokens:**
  - **S1:** `"Summer Lake, OR, 52998 31"`
  - **S3 (`S3-354627335`):** `"0052998 31, Oregon, Summer Lakke"`
  - **S2 (`S2-479094220`):** `"05037 CENTRAL AVENUE, PHOEIX, AZ"`

---

## 7. Multilingual and Indic Script Tokens in Addresses

State, district, or city names are frequently represented in regional Indic scripts (Devanagari, Gujarati, Kannada, etc.).

### Examples:
- **Hindi / Devanagari Script in Address (`महाराष्ट्र`, `मध्य प्रदेश`, `उत्तर प्रदेश`, `हरियाणा`):**
  - **S1:** `"Mumbai City, Maharashtra, Mumbai"`
  - **S3 (`S3-170854488`):** `"... Mumbai, Mumbai City, महाराष्ट्र"`
  - **S1:** `"125, Kanchan Bag, Ndore, Madhya Pradesh"`
  - **S2 (`S2-511813958`):** `"125, KANCHAN BAG, NDORE, मध्य प्रदेश"`
  - **S1:** `"Purwa Chetiya Bazaar, Siddharth Nagar, Uttar Pradesh"`
  - **S3 (`S3-232312120`):** `"उत्तर प्रदेश, Siddharth Nagar, ..."`
  - **S3 (`S3-890922145`):** `"Shop No C-29, Karnal, Karnal Alpha Intl City, हरियाणा"`
- **Gujarati Script in Address (`ગુજરાત`):**
  - **S1:** `"Survey No. 334/1, Halol, Panch Mahals, Gujarat"`
  - **S2 (`S2-67424467`):** `"#334/1, HALOL, PANCH MAHALS, ગુજરાત"`
- **Kannada Script in Address (`ಕರ್ನಾಟಕ`):**
  - **S2 (`S2-545337877`):** `"... PEENYA INDUSTRIAL AREA, ಕರ್ನಾಟಕ"`

---

## 8. Placeholder Values and Literal Null Indicators

External records contain string placeholders that must be scrubbed during normalization.

### Examples:
- **`NULL`, `<NULL>`, `nan`, `N/A`:**
  - **S3 (`S3-319489488`):** `"1129 Cherry Ridge Dr, NULL, Sugarcreek, Ohio"`
  - **S3 (`S3-948267524`):** `"83 Dehon Street, <NULL>, Revere, Massachusetts"`
  - **S2 (`S2-70188666`):** `"N/A, Tamil Nadu, 29, COONOOR ROAD"`
  - **S2 (`S2-778435241`):** `"NEEMUCH, NULL, MANASA, NO 58 C/O BHARAT KUMAR DHAKAD, Madhya Pradesh"`
