# Churn-Analysis
Telecom churn prediction project

# Retention IQ

A machine learning application for predicting telecom customer churn and helping retention officers identify customers who may need intervention.

The project combines a trained Random Forest model with a Streamlit dashboard, customer search and registry, customer-level risk signals, retention recommendations, analytics, and system logging.

# RetentionIQ

Telecom Customer Retention Intelligence — Group 2

This document explains the whole of `app.py`, top to bottom, in the order the code runs. I wrote it for everyone on the team, not only the people who read Python. If a section starts getting technical I say what the code does in plain language first, and put the mechanics after it.

If you only have five minutes, read "What this thing actually does" and "The numbers we quote, and where each one comes from". Those two sections cover everything you need to demo it or defend it.

## Contents

1. [What this thing actually does](#what-this-thing-actually-does)
2. [How to run it](#how-to-run-it)
3. [The file at a glance](#the-file-at-a-glance)
4. [Settings at the top of the file](#settings-at-the-top-of-the-file)
5. [Finding the model files](#finding-the-model-files)
6. [Theming: how the app handles light and dark](#theming-how-the-app-handles-light-and-dark)
7. [The database](#the-database)
8. [The model layer](#the-model-layer)
9. [How we explain a prediction](#how-we-explain-a-prediction)
10. [Startup: seeding and scoring](#startup-seeding-and-scoring)
11. [The portfolio layer](#the-portfolio-layer)
12. [The charts](#the-charts)
13. [Roles and navigation](#roles-and-navigation)
14. [Page by page](#page-by-page)
15. [The numbers we quote, and where each one comes from](#the-numbers-we-quote-and-where-each-one-comes-from)
16. [What this version cannot do](#what-this-version-cannot-do)
17. [How to change the things you will want to change](#how-to-change-the-things-you-will-want-to-change)

## What this thing actually does

We trained a model that predicts whether a telecom customer is about to leave. On its own that prediction is not worth much to a business. Knowing that a customer might leave does not tell you whether they are worth keeping, why they want to go, who should call them, or whether the call worked.

So the app is built around a loop rather than around the prediction:

```
Identify  ->  Score  ->  Explain  ->  Prioritise  ->  Intervene  ->  Track  ->  Measure
```

Read that as: find the customers, put a number on each one, work out why the number is high, decide who to call first, make an offer, record what happened, and count how many we saved. The model is only the second box. Most of the app is the other six.

There are six screens. Which ones you see depends on the role you pick in the sidebar.

**Executive Dashboard**: the money view. How many customers, how many at risk, what that costs per month, and the ten accounts to work today.

**Customer Registry**: the worklist. Every customer, sorted by what they are worth, with a filter for risk band and a button to open a case.

**Customer 360**: one customer at a time. Either look up someone already scored, or type in a new profile and score them. You get the risk band, the reasons, a costed list of what to offer, and the buttons that start the retention workflow.

**Retention Cases**: every intervention we have opened, what stage it is at, and how it ended.

**Analytics**: four tabs answering four questions: where are we losing customers, why, which ones should we save first, and did our interventions work.

**System Logs**: the audit trail. Administrator only, because a CEO should never have to look at it.

## How to run it

Locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens on `http://localhost:8501`.

Deployed, it runs on Streamlit Community Cloud from the `main` branch of this repo, with `app.py` as the main module.

What has to be present for it to work:

| File | Why |
|---|---|
| `app.py` | The whole application. One file. |
| `requirements.txt` | Package list. `scikit-learn` is pinned to `1.6.1` (see below). |
| `models/preprocessor.joblib` | Turns raw customer attributes into numbers the model understands. |
| `models/best_random_forest_tuned.pkl` | The trained random forest. |
| `prototype_seed_customers.csv` | The 70 starter customers. |
| `.streamlit/config.toml` | Theme colours and fonts. |

One warning that has already cost us a deploy: **`scikit-learn` must stay pinned at `1.6.1`**. That is the version the two model files were saved with. A pickled model is a frozen Python object, and loading it under a different scikit-learn version either throws an error or, worse, loads and behaves subtly differently. Do not relax that pin without re-training and re-saving the model.

## The file at a glance

`app.py` is about 2,400 lines. It is one file because Streamlit Cloud runs one main module and splitting it would mean managing imports across the repo for no real benefit at this size. It is organised into labelled blocks, each separated by a banner comment, and it runs strictly top to bottom.

| Lines | Block | What it is |
|---|---|---|
| 1–80 | Setup | Imports, product name, thresholds, file paths |
| 82–365 | Theming | Colour palettes, the stylesheet, small UI helpers |
| 367–648 | Database | Table definitions and every read and write |
| 650–952 | Model layer | Loading, scoring, explaining, the offer playbook |
| 954–1061 | Startup | Loading the seed data and scoring everyone once |
| 1063–1219 | Portfolio layer | The one table the whole app reads from |
| 1221–1437 | Charts | Every chart the app draws |
| 1440–1533 | Roles and navigation | Sidebar, masthead, role permissions |
| 1539–2433 | Pages | The six screens |

Something worth understanding before you read any further, because it explains a lot of the design decisions: **Streamlit re-runs this entire file from line 1 every time anyone touches anything.** Move a slider, click a tab, type a character, and the whole script executes again from the top. There is no "on click" handler like in a normal web app.

That is why the caching in this file matters so much. Without it, every keystroke would re-load the model from disk and re-score all 70 customers.

## Settings at the top of the file

```python
PRODUCT_NAME = "RetentionIQ"
PRODUCT_TAGLINE = "Telecom Customer Retention Intelligence"
BUILT_BY = "Developed by Group 2"

CHURN_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.60
RETENTION_HORIZON_MONTHS = 12
```

The three numbers are the ones you will get asked about.

`CHURN_THRESHOLD = 0.40` is the line above which we act. If the model says a customer has a 40% or higher chance of leaving, we flag them. Forty percent, not fifty, and that is deliberate: missing a customer who leaves costs us their whole future revenue, while flagging one who was going to stay costs a phone call. The two mistakes are not equally expensive, so the line does not sit in the middle. 

`HIGH_RISK_THRESHOLD = 0.60` splits the flagged customers into High and Medium. Everything below 40% is Low. So the three bands are:

- High: 60% and above
- Medium: 40% to 60%
- Low: below 40%

`RETENTION_HORIZON_MONTHS = 12` is used in one place only. On the Customer 360 screen, when we say "revenue protected if retained", we multiply the customer's monthly bill by twelve. It is an assumption, not a measurement, and the screen says so. Changing this one number changes every figure that depends on it.

## Finding the model files

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(*candidates):
    for candidate in candidates:
        full_path = os.path.join(BASE_DIR, candidate)
        if os.path.exists(full_path):
            return full_path
    return os.path.join(BASE_DIR, candidates[0])
```

This exists because of a real problem we hit. Our repo has both a `model/` folder and a `models/` folder, and the artifacts have moved between them. The original code looked in exactly one place and crashed when the files were somewhere else.

`find_file` takes a list of possible locations and returns the first one that exists. So the preprocessor is looked for at `models/preprocessor.joblib`, then `model/preprocessor.joblib`, then next to `app.py`. If none exist it returns the first path anyway, so the error message names a sensible location instead of saying nothing.

`BASE_DIR` is the folder `app.py` itself lives in. Everything is resolved relative to that rather than to the folder the app was launched from, which is what makes it work the same locally and on the server.

The seed data uses the same trick, and this is worth knowing:

```python
SEED_FILE = find_file(
    "data/telco_customer_churn.csv",
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "prototype_seed_customers.csv",
    "data/prototype_seed_customers.csv",
)
```

It looks for a full Telco dataset first and falls back to our 70-row sample. **If anyone drops the full 7,043-row CSV into `data/`, the app picks it up automatically and every number in the app scales to it.** No code change needed. That is the single highest-value thing left on the list.

## Theming: how the app handles light and dark

Streamlit lets each viewer choose light or dark from the settings menu. Our app draws a lot of its own HTML (the KPI tiles, the risk banner, the section headings), and those had hardcoded white text. A viewer in light mode got white text on a white background.

There is no clean CSS way to fix this. I checked: Streamlit exposes theme variables to custom components but not to the main page, so a stylesheet cannot read which theme is active. It has to be decided in Python.

```python
def active_palette():
    try:
        theme_type = st.context.theme.type
    except Exception:
        theme_type = None
    return LIGHT_PALETTE if theme_type == "light" else DARK_PALETTE
```

`st.context.theme.type` tells us which theme the viewer is actually looking at. We pick the matching palette and substitute its colours into the stylesheet. Anything unexpected falls back to dark, which is our configured default.

The two palettes hold the same set of names with different values: `text`, `heading`, `muted`, `surface`, `border`, chart colours and so on. Nothing in the file hardcodes a colour any more. If you want to change how the app looks, you edit those two dictionaries and nothing else.

The stylesheet uses `string.Template` rather than an f-string. CSS is full of curly braces and an f-string would need every single one doubled. `Template` only treats `$name` as a placeholder, so the CSS stays readable.

### The risk colours are different on purpose

```python
RISK_COLORS = {"High": "#d03b3b", "Medium": "#fab219", "Low": "#0ca30c"}
RISK_ICONS = {"High": "▲", "Medium": "◆", "Low": "●"}
```

Red, amber, green, and identical in both themes, unlike everything else. Status colours carry a fixed meaning, so they should not shift when the theme does.

The amber has a known weakness: on a near-white background it does not have enough contrast to meet accessibility standards. Rather than pick a different colour and lose the traffic-light meaning, every risk indicator in the app carries a shape and a word next to the colour. The distribution chart prints the count and percentage beside each bar. The risk banner spells out "MEDIUM RISK" in text. Nobody has to distinguish the colour to read the app, which is the point. Colour alone should never be the only carrier of meaning.

## The database

The app stores everything in a SQLite file called `telecom_churn.db`, created automatically on first run. SQLite is a database that lives in a single file with no server to install.

Four tables.

**`customers`**: one row per customer, holding the raw attributes the model needs: tenure, monthly charges, contract type, which services they have, and so on. It also records `actual_churn` (whether they historically left, where we know) and `source` (whether they came from the seed file or were scored in the app).

**`scans`**: one row per scoring event. Customer ID, the probability the model gave, the threshold in force at the time, and the resulting risk band. It is a history, not a snapshot: score the same customer twice and you get two rows. The app always reads the most recent one.

**`interventions`**: the retention workflow, and the table that makes this a product rather than a report. One row per case:

```sql
CREATE TABLE IF NOT EXISTS interventions (
    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customerID TEXT,
    opened_at TEXT,
    updated_at TEXT,
    risk_level TEXT,
    churn_probability REAL,
    monthly_value REAL,
    main_driver TEXT,
    assigned_to TEXT,
    status TEXT,
    offer_type TEXT,
    contact_attempted INTEGER DEFAULT 0,
    offer_made INTEGER DEFAULT 0,
    customer_accepted INTEGER DEFAULT 0,
    notes TEXT
)
```

Notice that it stores the risk level and probability *at the moment the case was opened*, not a link to the current score. That is on purpose. Six months later you want to know what we believed when we decided to act, not what we believe now. Without it you cannot honestly measure whether the model was right.

A case moves through five states: Open, Contacted, Offer Made, and then either Retained or Lost.

**`system_logs`**: timestamped record of what the app did. Seeding, scoring, case changes, and any failure to load the model.

### One pattern worth explaining

Nearly every write function looks like this:

```python
def log_event(level, event, customer_id="", details="", connection=None):
    owns_connection = connection is None
    if owns_connection:
        connection = get_connection()

    connection.execute(...)

    if owns_connection:
        connection.commit()
        connection.close()
```

The `connection=None` argument is a performance fix, and it made a real difference. Every `commit()` forces the operating system to flush to disk, which is slow, especially on a cloud server with network-backed storage. The original code opened a connection, wrote one row, committed and closed, once per customer. Starting the app meant roughly 280 of those cycles.

Now a caller that is doing many writes opens one connection, passes it in, and commits once at the end. When no connection is passed the function manages its own, so single writes still work unchanged. Startup went from around 280 disk flushes to two.

### The double-click guard

```python
def open_case(customer_id, ...):
    existing = get_open_case(customer_id)
    if existing and existing["status"] not in ("Retained", "Lost"):
        return existing["case_id"], False
    ...
    return case_id, True
```

If a customer already has a case that has not been closed, we return the existing one instead of creating a second. The function reports which happened, so the screen can say "already has an active case" rather than silently doing nothing. Without this, an impatient double-click during the demo would put duplicate work in the queue. we tested it: clicking twice produces one case.

## The model layer

`load_artifacts()` reads the two files from disk. It carries `@st.cache_resource`, which means Streamlit runs it once and hands back the same objects on every later re-run. Without it the model would be re-read from disk on every keystroke.

`transform_features()` fixes a bug that was flooding our deployment log with thousands of warnings:

```python
def transform_features(input_df, preprocessor):
    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out())
```

The preprocessor hands back a plain grid of numbers with no column names. The model was trained on a table that had names. Every prediction therefore printed `X does not have valid feature names`. The predictions were correct (the warning is cosmetic), but 70 of them appeared at startup and it made the log unreadable. Re-attaching the names silences it. Same numbers out, no noise.

`score_frame()` scores any number of customers in a single call. This matters more than it sounds. Scoring 70 customers one at a time is 70 separate calls into the model; scoring them as one table is one call. On the full 7,043-row dataset the difference is minutes versus about a second.

`risk_band()` turns a probability into High, Medium or Low using the two thresholds.

## How we explain a prediction

We explain predictions two different ways, because the two situations have different constraints.

### Across the whole portfolio: the model's own importances

```python
@st.cache_data(show_spinner=False)
def global_churn_drivers(_model, _preprocessor):
    importances = _model.feature_importances_
    names = list(_preprocessor.get_feature_names_out())
    ...
```

A random forest can report how much each input mattered to its decisions. The problem is that it reports on the *transformed* inputs, not the human ones. "Contract type" was split during preprocessing into three separate yes/no columns, one for month-to-month, one for one year and one for two year, so the model reports three numbers where the business wants one.

This function walks every transformed column, works out which original attribute it came from, and adds the importances back together. The output is the Top Churn Drivers chart: Contract type 24.2%, Tenure 12.5%, Online security 9.5%, and so on.

Those percentages come from the trained model. We did not write them down and we cannot tune them. If someone asks whether the drivers are the model's or ours, they are the model's.

### For one customer: the sensitivity analysis

```python
def get_customer_signals(input_df, preprocessor, model):
    base_probability = get_churn_probability(input_df, preprocessor, model)
    for feature in MODEL_FEATURES:
        # try realistic alternative values, re-score, keep the biggest swing
```

For a single customer we do something better. We take their profile, change one attribute, score them again, and see how much the probability moved. Do that across every attribute and you learn which change would help this specific person most.

That produces the table on Customer 360, the one that says something like "Contract type: currently Month-to-month, best alternative Two year, would reduce risk by 22%". A retention agent can read that as a script for the call.

It costs roughly 40 model calls per customer. Fine for one person on one screen.

### Why the registry uses a shortcut instead

40 calls per customer is fine for one. For 7,043 it is about 280,000 calls, which takes minutes and would make the registry unusable.

So the registry uses a rule ladder:

```python
DRIVER_RULES = [
    ("Contract", lambda c: c.get("Contract") == "Month-to-month", "Month-to-month contract"),
    ("tenure", lambda c: float(c.get("tenure") or 0) < 12, "Low tenure (under 12 months)"),
    ("InternetService", lambda c: c.get("InternetService") == "Fiber optic", "Fiber optic service"),
    ...
]
```

Each rule is a risk condition and a label. `build_driver_ladder()` sorts them by the model's global importance for that attribute, so the ladder is ordered by what the model actually cares about, not by what we assume. `primary_driver()` walks down and returns the first condition this customer meets.

The Main Driver column is a rule, not a model output. It is *ordered* by the model, which is why it is defensible, but it is a fast approximation of the expensive analysis. The honest one-line version is "ranked by the model's own importances, filtered to the conditions this customer meets".

### The playbook

`retention_playbook()` turns a customer's profile into specific actions with costs attached. Month-to-month gets an annual contract offer at roughly 10% of their monthly bill. No tech support gets three free months at about 5%. High monthly charges get a plan review with a 15% loyalty discount. Below the threshold, the only recommendation is to keep monitoring.

The costs are planning assumptions expressed as a share of the customer's bill, and the screen says exactly that. They are not quoted commercial terms and we should not present them as such.

## Startup: seeding and scoring

Three things happen the first time anyone opens the app.

`seed_customer_registry()` reads the CSV and loads the customers. It handles one quirk of the published Telco dataset: `TotalCharges` is stored as text and is blank for brand-new accounts, which would reach the model as the string "NaN". We convert it to a number and fill blanks with zero. All 70 inserts happen inside one transaction.

`generate_seed_scans()` scores everyone. It first asks the database which customers already have a score, so re-running does not duplicate work. Then it scores all the remaining ones in a single call and writes the results in one transaction.

`bootstrap()` wraps all of it:

```python
@st.cache_resource(show_spinner="Scoring the customer portfolio...")
def bootstrap():
    initialize_database()
    seed_customer_registry()
    try:
        preprocessor, model = load_artifacts()
        generate_seed_scans(preprocessor, model)
        return preprocessor, model, True, ""
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        log_event("ERROR", "Model artifacts could not be loaded", details=message)
        return None, None, False, message
```

`@st.cache_resource` is doing the heavy lifting. Because Streamlit re-runs the file on every interaction, without this decorator every click would re-seed and re-score. With it, the work happens once per server and every later run gets the cached result back instantly. Cold start is about 2.5 seconds; a click after that is under a tenth of a second.

The `try`/`except` means a missing or unreadable model file produces a clear error screen naming the paths we looked in, instead of a stack trace.

## The portfolio layer

Every screen reads from one table, built once by `portfolio()`. It joins each customer to their most recent score and their most recent case, then adds the columns the business cares about:

- `monthly_value`: their monthly bill
- `expected_loss`: monthly bill multiplied by churn probability
- `at_risk`: true when probability is at or above 40%
- `main_driver`: from the rule ladder
- `tenure_band`: 0–6, 7–12, 13–24, 25–48, 49+ months
- `charge_band`: under $35, $35–60, $60–80, $80–100, $100+
- `segment`: High value / loyal, High value / new, Standard / loyal, Standard / new
- `status`: the case status, or "No case"

The bands exist so the analytics charts have something to group by. A chart of churn against raw monthly charges is a cloud of dots; a chart against five price bands answers a question.

`expected_loss` is the most important column in the file, and it is the one that turns a model into a business tool. A customer with a 90% chance of leaving and a $20 bill has an expected loss of $18. A customer with a 75% chance and a $300 bill has an expected loss of $225. The second customer is more than ten times more urgent even though the model thinks they are *less* likely to leave. The registry sorts by this by default.

### The caching token

```python
def data_version():
    scans = connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    cases = connection.execute("SELECT COUNT(*) FROM interventions").fetchone()[0]
    touched = connection.execute("SELECT COALESCE(MAX(updated_at), '') FROM interventions").fetchone()[0]
    return f"{scans}-{cases}-{touched}"
```

This solves a small problem with a neat trick. We want the portfolio table cached so screens are fast, but cached data goes stale. Open a case and the registry would still show the old status.

So we build a short string from the number of scans, the number of cases, and the timestamp of the most recent case change, and pass it into the cached function as an argument. Any write changes the string. A changed argument means Streamlit treats it as a new call and rebuilds. Nothing changes, and the cache is reused.

Three fast counting queries buy us the whole cache.

## The charts

Every chart is drawn with Altair, which already ships with Streamlit, so no new dependency.

The colour choices follow one rule: a chart with a single series uses one colour and no legend, because the title already says what it is. Multi-colour is reserved for when colour actually carries meaning, which in this app is only the risk bands.

Every chart has value labels printed directly on it. Partly that is the accessibility mitigation for the amber, but mostly it is because a presentation audience cannot hover a tooltip and should not have to estimate a bar against an axis.

`style_chart()` applies the theme to every chart in one place: axis colours, grid colours, transparent background so the app's gradient shows through. Change it once, every chart follows.

The specific charts:

`risk_rate_chart()`: the share of customers at risk within each category. Used for contract type, payment method, internet service, charge band and segment.

`revenue_chart()`: expected monthly loss by category. Same grouping, different question: not where the customers are, but where the money is.

`risk_distribution_chart()`: High, Medium and Low counts using the fixed risk colours, with counts and percentages printed beside each bar.

`drivers_chart()`: the model's importances, biggest first.

`lifecycle_chart()`: risk against tenure band, drawn as a filled line. This one usually gets a reaction, because it shows plainly that risk concentrates in the first year.

`sample_note()`: prints an honesty caption when the registry has fewer than 2,000 customers, saying the segment cells are small and the results are directional.

## Roles and navigation

```python
ROLE_PAGES = {
    "Executive": ["Executive Dashboard", "Analytics"],
    "Retention Manager": ["Executive Dashboard", "Customer Registry", "Retention Cases", "Analytics"],
    "Retention Agent": ["Customer Registry", "Customer 360", "Retention Cases"],
    "Data Analyst": ["Executive Dashboard", "Analytics", "Customer Registry", "Customer 360"],
    "Administrator": [everything, including "System Logs"],
}
```

Pick a role in the sidebar and the navigation only offers the pages that role uses. An executive gets the money view and the analysis. An agent gets their worklist and the customer screens. Only the administrator sees System Logs.

**This is a demonstration of role-based views, not security.** Anyone can switch role from the dropdown. There is no login and no password. 

The point it demonstrates is a product one. Different people need different screens, and a chief customer officer should never be scrolling past a system log to find the revenue number.

## Page by page

### Executive Dashboard

Six KPI tiles across two rows: total customers, customers at risk, churn risk rate, monthly revenue at risk, risk-adjusted exposure, revenue protected.

Under them, a paragraph explaining the difference between the two revenue figures, because they are different numbers and someone will ask.

Below that, a warning that appears only when the registry's known churn rate is above 40%. With our 70-row sample it always appears, and it says the sample is deliberately balanced, that the real IBM Telco dataset churns at about 27%, and that our percentages are therefore not population estimates.

Then four charts (risk distribution, churn by contract, top drivers, revenue at risk by contract) and an "Act today" table of the ten accounts with the highest expected loss.

### Customer Registry

Filters across the top: risk band, case status, sort field, and an ID search. Four summary tiles that recalculate for whatever the filters currently show. Then the worklist itself.

The columns are: Customer, Churn probability, Risk level, Monthly value, Expected loss, Main driver, Intervention status, Owner. Probability renders as a progress bar rather than a number, so you can scan the column instead of reading it.

At the bottom, pick a customer, an owner and an opening offer, and open a case.

### Customer 360

Two tabs. One looks up a customer we have already scored, ordered by expected loss. The other takes a fresh profile.

The scoring form has nineteen inputs, arranged in three columns under headings for account details, services, and household. The submit button sits at the top next to the customer ID rather than below all nineteen fields. 

Either way you arrive at the same result card, drawn by `render_result()`:

A coloured risk banner with the band, the customer ID and the probability at large size. Four figures: monthly value, expected monthly loss, revenue protected if retained, estimated offer cost. The costed playbook, each action with its reasoning. The sensitivity table showing which single change would help most. Then four workflow buttons (Assign to Retention, Create Case, Contact Customer, Mark Intervention), each writing to the interventions table.

One implementation detail that caused a bug worth remembering: Streamlit renders both tabs even when only one is visible, so `render_result` runs twice on every load. Two buttons with the same identifier crash the app. The function takes a `key_prefix` argument and the two tabs pass different values, which keeps them distinct.

### Retention Cases

Four tiles at the top: cases opened, in progress, retained, and monthly revenue saved.

A pipeline chart showing how many cases sit at each stage. The full case book, with the risk level and probability recorded at the time each case was opened. And a form to update a case: change the status, reassign the owner, record the offer, and mark whether the customer accepted.

Move a case to Retained and their monthly value flows straight through to the Revenue Protected tile on the executive dashboard. I tested that path end to end. Opening a case for the highest-value at-risk customer and closing it as Retained moved that tile from $0 to $110.

### Analytics

Four tabs, each named for the question it answers.

**Where**: churn risk by contract, payment method, internet service, charge band, and across the lifecycle.

**Why**: the model's driver importances, and a count of how many at-risk customers each primary driver accounts for.

**Who**: risk and expected loss by segment, expected loss across the lifecycle, and the fifteen individual accounts worth a personal call.

**Did it work**: campaign performance, offer outcomes, and the model's accuracy on customers whose real outcome we know. That last panel also shows recall, which is the share of customers who actually left that we caught. Recall matters more than accuracy here, and the threshold is set the way it is to protect it.

Every chart has a caption naming the business question. That was deliberate. A chart that does not answer a question is decoration.

### System Logs

Event count, error count, which seed file the registry came from, and the last 500 log lines. Administrator only.

## The numbers we quote, and where each one comes from


| Number | Formula | Watch out for |
|---|---|---|
| Total customers | Row count in `customers` | 70 today. Scales automatically with a bigger seed file. |
| Customers at risk | Count where probability ≥ 40% | Uses the action threshold, not the High band. |
| Churn risk rate | At risk ÷ total | Reads about 60% on our sample because the sample is balanced. |
| Monthly revenue at risk | Sum of monthly charges for at-risk customers | Gross. Assumes every flagged customer leaves, which they will not. |
| Risk-adjusted exposure | Sum of (monthly charges × probability) across everyone | The defensible one. This is the figure to plan against. |
| Expected loss per customer | Monthly charges × probability | Drives the sort order everywhere. |
| Revenue protected | Sum of monthly charges for customers whose case status is Retained | Only counts cases actually closed as Retained. |
| Win rate | Retained ÷ (retained + lost) | Undefined until at least one case is resolved. |
| Revenue protected if retained | Monthly charges × 12 | An assumption about horizon, labelled as one on screen. |
| Offer cost | A percentage of the customer's monthly bill per action | A planning assumption. Not a real commercial rate. |
| Accuracy | Correct predictions ÷ customers with a known outcome | Only historical customers have a known outcome. |
| Recall | Caught churners ÷ actual churners | The more important of the two. |

The difference between the two revenue figures: gross revenue at risk assumes everyone flagged leaves, and risk-adjusted exposure weights each customer by how likely they actually are to go. We show both because the gap between them is itself information, and because the second is the one a finance team would accept.

## What this version cannot do


**The registry is 70 customers, and the sample is balanced.** Roughly half of the seed customers historically churned, so the app reports about a 60% risk rate. A real customer base does not look like that. The published Telco population is around 27%. The dashboard says this in a warning banner. Dropping the full dataset into `data/` fixes it completely with no code change.

**The database resets.** Streamlit Community Cloud gives each app a temporary filesystem. Cases we open during the demo survive until the container restarts and then the app re-seeds from the CSV. Fine for a presentation, not fine for real use, and the Customer 360 screen says so. Production would need a hosted database.

**Role switching is not security.** Covered above. It shapes what the product shows; it protects nothing.

**Main Driver in the registry is a rule, not a model output.** The rules are ordered by the model's importances, but the expensive per-customer analysis only runs on Customer 360.

**Offer costs are invented.** They are percentages of the customer's bill, chosen to be plausible. Real numbers would come from finance.

**There is no time series.** Every seed customer was scored in the same instant, so we cannot show risk trending over weeks. The lifecycle chart shows risk against tenure instead, which answers a similar question honestly.

## How to change the things you will want to change

**Change the decision threshold.** Edit `CHURN_THRESHOLD` at the top. Everything downstream follows: the flags, the KPIs, the charts, the bands.

**Change the colours.** Edit `DARK_PALETTE` and `LIGHT_PALETTE`. Both hold the same keys. No colour is hardcoded anywhere else.

**Change the fonts or the default theme.** That is `.streamlit/config.toml`, not `app.py`.

**Add a retention officer or an offer type.** Append to `RETENTION_OFFICERS` or `OFFER_TYPES` near the top of the database section. Both dropdowns read from those lists.

**Add a new chart.** Write a function that returns an Altair chart, wrap it in `style_chart()`, and call it from a page. Copy `risk_rate_chart` as a starting point.

**Add a page.** Add the branch at the bottom following the `elif page == "..."` pattern, then add the page name to whichever roles should see it in `ROLE_PAGES`.

**Change who sees what.** `ROLE_PAGES` only.

**Load the full dataset.** Put the CSV at `data/telco_customer_churn.csv` and delete `telecom_churn.db` if you are running locally. Everything scales on its own.

**Re-train the model.** Save the new preprocessor and model into `models/`, and update the `scikit-learn` pin in `requirements.txt` to whatever version you trained with. Those two things have to match.

The application is deployed and ready for demonstration and final project submission.
