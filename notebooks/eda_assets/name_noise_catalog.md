# Business Name Noise Catalog

This catalog documents the empirical noise patterns observed across the training records when comparing Source 1 (canonical reference) to true matching records in Source 2 and Source 3.

---

## 1. Legal Entity Suffix Variations & Normalization

Legal suffixes are frequently expanded, abbreviated, omitted, or placed in parentheses/brackets across data sources.

### Examples:
- **`Limited` vs `Ltd` vs `(Ltd)`:**
  - **S1:** `"Services Ten Agro Limited"`
  - **S2 (`S2-78586487`):** `"Services Ten Agro Ltd"`
  - **S1:** `"Sanskruti Technologiesprivate Pvt Ltd"`
  - **S2 (`S2-70188666`):** `"Sanskruti Technologiesprivate Pvt (Ltd)"`
- **`Private Limited` vs `Private` vs `Pvt Ltd`:**
  - **S1:** `"Bombay Investment Private Limited"`
  - **S3 (`S3-752044877`):** `"Bombay Investment Private"`
  - **S1:** `"Naidu Balaji Private Limited"`
  - **S2 (`S2-521019011`):** `"NAlDU BALAJI PRIVATE"`
- **`LLC` Dropped or Appended:**
  - **S1:** `"2598 Center Street Management LLC"`
  - **S3 (`S3-712837153`):** `"2598 Center Street Management"`
  - **S1:** `"Spears, Philippe, L.C.S.W."`
  - **S2 (`S2-646039812`):** `"Spears, Philippe, L.C.S.W. [LLC]"`
- **`Corp` / `Corporation` / `Inc`:**
  - **S1:** `"Basalt Corp"`
  - **S2 (`S2-53015605`):** `"Basalt"`
  - **S1:** `"Olanola Allstate Clinic"`
  - **S2 (`S2-937091799`):** `"Olanola Allstate Clinic Ltd"`

---

## 2. Punctuation & Conjunction Variations ("&" vs "and" vs "+")

Conjunctions and punctuation marks differ significantly across sources, including ampersands, plus signs, slashes, and omission.

### Examples:
- **`&` vs `+` vs `and`:**
  - **S1:** `"Willetts, Curry & Crossman Clinic"`
  - **S2 (`S2-857629781`):** `"Willetts, Curry + Crossman ([Clinic])"`
- **Punctuation Omission & Spacing:**
  - **S1:** `"Valencia, Knicely & Prado"`
  - **S3 (`S3-70790379`):** `"Valencia Knicely & Prado"`
- **Exclamation & Special Characters:**
  - **S1:** `"Uptown Bakery!"`
  - **S2 (`S2-492276871`):** `"Uptown Bakery!"`

---

## 3. Abbreviations vs Full Words in Core Name

Words within the primary business identity are often contracted or colloquialized.

### Examples:
- **`Technologiesprivate` vs `Technologies` / `Tech`:**
  - **S1:** `"Sanskruti Technologiesprivate Pvt Ltd"`
  - **S2 (`S2-70188666`):** `"Sanskruti Technologiesprivate Pvt (Ltd)"`
- **`Advisors` vs `Advisos` / `Consultants` vs `Services`:**
  - **S1:** `"Rajnagar Consultants"`
  - **S3 (`S3-530137928`):** `"Rajnagar Services"`
- **`Solutions Al Spaces Center` vs `Spaces Center`:**
  - **S1:** `"Solutions Al Spaces Center"`
  - **S3 (`S3-728059936`):** `"5olutions Al Spaces Center"`

---

## 4. Word-Order Permutations & Inversions

Component tokens within the name are frequently transposed, such as entity types prepended or trade descriptors placed first.

### Examples:
- **Legal Suffix / Prefix Transposition:**
  - **S1:** `"Peridos Investment LLC"`
  - **S2 (`S2-440166852`):** `"LLC Peridos Ínvestment"`
- **Descriptor Reordering:**
  - **S1:** `"Ss International Private Limited"`
  - **S3 (`S3-170854488`):** `"Shri Ss International Limited Private"`
- **Complex Token Scrambling:**
  - **S1:** `"Limited Novel Of Technology Private Center"`
  - **S3 (`S3-491935776`):** `"Limited Novel Of Technology Private Center"` (vs `"Novel Technology Pvt Ltd"`)

---

## 5. Typos, OCR Errors, and Character Perturbations

Character errors include digit-for-letter substitution (leetspeak/OCR), accent diacritics, and phonetic misspellings.

### Examples:
- **Digit / Character OCR Substitutions (`5` for `S`, `6` for `G`, `l` for `I`):**
  - **S1:** `"Solutions Al Spaces Center"`
  - **S3 (`S3-728059936`):** `"5olutions Al Spaces Center"`
  - **S1:** `"RK Great Advisors LLC"`
  - **S2 (`S2-944913305`):** `"RK 6reat Advisors LLC"`
- **Accent Diacritics Inserted:**
  - **S1:** `"Bethina Hodge Bay Colombier LLC"`
  - **S2 (`S2-414630023`):** `"BETHINA HODGE BÁY COLOMBIER LLC"`
  - **S1:** `"Smart Trailblazer LLC"`
  - **S3 (`S3-825835848`):** `"Smart Tráilblazer LLC"`
  - **S1:** `"Peridos Investment LLC"`
  - **S2 (`S2-440166852`):** `"LLC Peridos Ínvestment"`
- **Character Transpositions / Misspellings:**
  - **S1:** `"Kelci's Reliable Realty"`
  - **S3 (`S3-854776208`):** `"Kelci'S Rleiame Realty"`
  - **S1:** `"Mahathi Enterprises Private Limited"`
  - **S3 (`S3-814030104`):** `"Mahathi Entecpries Private Limited"`
  - **S1:** `"Willetts, Curry & Crossman Clinic"`
  - **S2 (`S2-514191558`):** `"Willetts, Curry & Crotsmna Clinic"`

---

## 6. Transliteration Differences (Non-Latin Indic Scripts)

In the Indian subset, names frequently appear in regional Indic scripts (Devanagari/Hindi, Gujarati, Tamil, etc.) rather than Latin English transliterations.

### Examples:
- **Devanagari (Hindi) Script vs Latin:**
  - **S1:** `"Great Engineering Limited"`
  - **S3 (`S3-286798794`):** `"ग्रेट इंजीनियरिंग लिमिटेड"`
  - **S1:** `"Galaxy Finance Private Limited"`
  - **S2 (`S2-175745851`):** `"गैलेक्सी फाइनेंस प्राइवेट लिमिटेड"`
  - **S1:** `"Prime Trading"`
  - **S2 (`S2-511813958`):** `"प्राइम ट्रेडिंग"`
- **Gujarati Script vs Latin:**
  - **S1:** `"My Power Private Limited"`
  - **S2 (`S2-67424467`):** `"માય પાવર પ્રાઇવેટ લિમિટેડ"`
- **Tamil Script vs Latin:**
  - **S1:** `"Sree Power Pvt Ltd"`
  - **S2 (`S2-108393454`):** `"ஸ்ரீ பவர் பிரைவேட் லிமிடெட்"`

---

## 7. Casing Inconsistencies

Extreme casing variations exist across all three sources.

### Examples:
- **ALL CAPS vs Title Case:**
  - **S1:** `"XTG Enterprises Private Limited"`
  - **S3 (`S3-407428181`):** `"XTG ENTERPRISES PRIVATE LIMITED"`
  - **S1:** `"Urgent Care Empire Clinic"`
  - **S3 (`S3-23005351`):** `"URGENT CARE EMPIRE CLINIC"`
- **all lowercase vs Title Case:**
  - **S1:** `"Williams Construction"`
  - **S3 (`S3-645104940`):** `"williams construction"`

---

## 8. Web URLs, Handles, DBA Tags, and Extra Appended Words

External records (S2/S3) frequently contain website URLs, social media handles, honorific prefixes, or descriptive suffixes.

### Examples:
- **Domain Names / URLs / Social Handles in Name Field:**
  - **S1:** `"Solora Wave LLC"`
  - **S3 (`S3-319489488`):** `"wavesolora.com"`
  - **S1:** `"Quality Asset Solutions, LLC"`
  - **S3 (`S3-727461540`):** `"@qualityasset"`
  - **S1:** `"Signature Marketing Alliance LLC"`
  - **S2 (`S2-77841735`):** `"signaturemarketingalliance.com"`
  - **S1:** `"Lakey, Harvey & Robertson Alpex LP"`
  - **S3 (`S3-354627335`):** `"Lakeyharveyrobertson.Com"`
- **Honorific Prefixes (`Shri`, `Mr`):**
  - **S1:** `"Ss International Private Limited"`
  - **S3 (`S3-170854488`):** `"Shri Ss International Limited Private"`
  - **S1:** `"Prime Trading"`
  - **S3 (`S3-464218104`):** `"Mr Prime Center"`
- **Appended Descriptors (`Partners`, `Center`, `Association`, `née`):**
  - **S1:** `"Signature Marketing Alliance LLC"`
  - **S3 (`S3-622985367`):** `"Signature Alliance LLC Partners"`
  - **S1:** `"Faus Inc"`
  - **S3 (`S3-166649845`):** `"Vantagetavo née Faus Inc"`
- **Word Duplication / Glitches:**
  - **S1:** `"Slate Premier Inc"`
  - **S2 (`S2-404216086`):** `"SLATE-PENMIIR SLATE-PENMIIR INC"`
