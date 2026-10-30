"""
Step 2 — Exploratory Data Analysis (answers the 20 questions in the brief)
Saves one PNG per question to reports/eda/ and a text summary to
reports/eda_summary.md

Run:  python src/eda.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed_housing.csv"
OUT = ROOT / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="crest")
notes = []


def save(fig, n, title, finding):
    fig.suptitle(f"Q{n}. {title}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"q{n:02d}.png", dpi=110)
    plt.close(fig)
    notes.append(f"**Q{n}. {title}**  \n{finding}\n")
    print(f"Q{n} done")


df = pd.read_csv(DATA)
ppsf = "Price_per_SqFt"

# ---------------- 1–5 Price & Size
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["Price_in_Lakhs"], bins=50, ax=ax)
ax.set_xlabel("Price (₹ Lakhs)")
save(fig, 1, "Distribution of property prices",
     f"Prices range ₹{df.Price_in_Lakhs.min():.0f}–{df.Price_in_Lakhs.max():.0f} L, "
     f"median ₹{df.Price_in_Lakhs.median():.0f} L; distribution is close to uniform (no strong skew).")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["Size_in_SqFt"], bins=50, ax=ax)
save(fig, 2, "Distribution of property sizes",
     f"Sizes range {df.Size_in_SqFt.min()}–{df.Size_in_SqFt.max()} sq ft, median {df.Size_in_SqFt.median():.0f}; roughly uniform.")

fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="Property_Type", y=ppsf, showfliers=False, ax=ax)
m = df.groupby("Property_Type")[ppsf].median().round(0)
save(fig, 3, "Price per sq ft by property type",
     "Median ₹/sqft — " + ", ".join(f"{k}: ₹{v:,.0f}" for k, v in m.items()) + ". Type barely moves price/sqft.")

samp = df.sample(5000, random_state=1)
fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(data=samp, x="Size_in_SqFt", y="Price_in_Lakhs", s=8, alpha=.4, ax=ax)
r = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
save(fig, 4, "Size vs price", f"Pearson r = {r:.3f}: size and total price are essentially uncorrelated in this dataset.")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df[ppsf], ax=axes[0]); axes[0].set_title("Price per sq ft")
sns.boxplot(y=df["Size_in_SqFt"], ax=axes[1]); axes[1].set_title("Size")
save(fig, 5, "Outliers in price/sqft and size",
     f"{df.PPSF_Outlier.sum():,} rows ({df.PPSF_Outlier.mean():.1%}) are IQR outliers on price/sqft "
     "(small homes with high prices); size has no outliers. Outliers are flagged, not removed.")

# ---------------- 6–10 Location
s = df.groupby("State")[ppsf].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
s.plot.barh(ax=ax); ax.set_xlabel("Avg ₹/sqft")
save(fig, 6, "Average price per sq ft by state",
     f"Highest: {s.index[-1]} (₹{s.iloc[-1]:,.0f}); lowest: {s.index[0]} (₹{s.iloc[0]:,.0f}). Spread is narrow (~{(s.max()/s.min()-1):.0%}).")

c = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 9))
c.plot.barh(ax=ax); ax.set_xlabel("Avg price (₹ L)")
save(fig, 7, "Average property price by city",
     f"Most expensive: {c.index[-1]} (₹{c.iloc[-1]:.1f} L); cheapest: {c.index[0]} (₹{c.iloc[0]:.1f} L).")

age = df.groupby("Locality")["Age_of_Property"].median().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(age, bins=20, ax=ax); ax.set_xlabel("Median age (years) per locality")
save(fig, 8, "Median property age by locality",
     f"Across 500 localities median age ranges {age.min():.0f}–{age.max():.0f} yrs; oldest stock: {', '.join(age.index[:3])}.")

top_c = df["City"].value_counts().index[:12]
ct = pd.crosstab(df[df.City.isin(top_c)]["City"], df["BHK"], normalize="index")
fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(ct, annot=True, fmt=".0%", cmap="crest", ax=ax)
save(fig, 9, "BHK distribution across cities (top 12 by listings)",
     "1–5 BHK are evenly represented (~20% each) in every city.")

top_loc = df.groupby("Locality")[ppsf].mean().nlargest(5).index
tr = df[df.Locality.isin(top_loc)].groupby(["Year_Built", "Locality"])["Price_in_Lakhs"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=tr, x="Year_Built", y="Price_in_Lakhs", hue="Locality", ax=ax)
save(fig, 10, "Price trend by build year — top 5 most expensive localities",
     f"Top localities by ₹/sqft: {', '.join(top_loc)}. Price vs build year is flat/noisy — no clear vintage premium.")

# ---------------- 11–15 Correlation
num = ["Price_in_Lakhs", "Size_in_SqFt", ppsf, "BHK", "Age_of_Property", "Nearby_Schools",
       "Nearby_Hospitals", "Floor_No", "Total_Floors", "Amenity_Count", "Infrastructure_Score",
       "Annual_Growth_Rate", "Future_Price_5Y"]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
save(fig, 11, "Correlation between numeric features",
     "Raw features are almost independent of each other (|r|<0.05). Strong links are only engineered ones: "
     "Price ↔ Future price, and Price/sqft ↔ Size (inverse).")

for n, col in [(12, "Nearby_Schools"), (13, "Nearby_Hospitals")]:
    g = df.groupby(col)[ppsf].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    g.plot(marker="o", ax=ax); ax.set_ylabel("Avg ₹/sqft")
    save(fig, n, f"{col.replace('_', ' ')} vs price per sq ft",
         f"Avg ₹/sqft stays within ₹{g.min():,.0f}–{g.max():,.0f} across 1–10 {col.split('_')[1].lower()}: no meaningful effect on current price "
         "(but it raises the projected growth rate).")

fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", ax=ax)
f = df.groupby("Furnished_Status")["Price_in_Lakhs"].median()
save(fig, 14, "Price by furnished status",
     "Median price — " + ", ".join(f"{k}: ₹{v:.0f} L" for k, v in f.items()) + ".")

fc = df.groupby("Facing")[ppsf].mean()
fig, ax = plt.subplots(figsize=(8, 4))
fc.plot.bar(ax=ax, rot=0); ax.set_ylabel("Avg ₹/sqft")
save(fig, 15, "Price per sq ft by facing direction",
     "Avg ₹/sqft — " + ", ".join(f"{k}: ₹{v:,.0f}" for k, v in fc.items()) + ". Facing has no material effect.")

# ---------------- 16–20 Investment / amenities / ownership
for n, col in [(16, "Owner_Type"), (17, "Availability_Status")]:
    vc = df[col].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    vc.plot.bar(ax=ax, rot=0)
    save(fig, n, f"Properties by {col.replace('_', ' ').lower()}",
         ", ".join(f"{k}: {v:,}" for k, v in vc.items()) + ".")

fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df, x="Parking_Space", y="Price_in_Lakhs", ax=ax)
p = df.groupby("Parking_Space")["Price_in_Lakhs"].mean()
save(fig, 18, "Does parking affect price?",
     f"Avg price with parking ₹{p['Yes']:.1f} L vs without ₹{p['No']:.1f} L — negligible difference.")

am = df.groupby("Amenity_Count")[ppsf].mean()
amg = df.groupby("Amenity_Count")["Good_Investment"].mean()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
am.plot.bar(ax=ax[0], rot=0); ax[0].set_ylabel("Avg ₹/sqft")
amg.plot.bar(ax=ax[1], rot=0, color="seagreen"); ax[1].set_ylabel("% Good Investment")
save(fig, 19, "Amenities vs price per sq ft and investment quality",
     f"Amenity count doesn't change ₹/sqft, but Good-Investment share rises from {amg.min():.0%} to {amg.max():.0%}.")

tp = df.groupby("Public_Transport_Accessibility").agg(ppsf=(ppsf, "mean"),
                                                      good=("Good_Investment", "mean"),
                                                      growth=("Annual_Growth_Rate", "mean")).loc[["Low", "Medium", "High"]]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
tp["ppsf"].plot.bar(ax=ax[0], rot=0); ax[0].set_ylabel("Avg ₹/sqft")
tp["good"].plot.bar(ax=ax[1], rot=0, color="seagreen"); ax[1].set_ylabel("% Good Investment")
save(fig, 20, "Public transport accessibility vs price & investment potential",
     f"₹/sqft is flat, but Good-Investment share goes Low {tp.good['Low']:.0%} → High {tp.good['High']:.0%} "
     f"and avg growth {tp.growth['Low']:.1%} → {tp.growth['High']:.1%}.")

# Extra brief items: crime-rate proxy & infrastructure vs resale value
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="Good_Investment", y="Infrastructure_Score", ax=ax)
save(fig, 21, "Infrastructure score vs Good Investment / resale value",
     f"Corr(Infrastructure, Future price) = {df.Infrastructure_Score.corr(df.Future_Price_5Y):.3f}; "
     f"Corr(Infrastructure, growth rate) = {df.Infrastructure_Score.corr(df.Annual_Growth_Rate):.3f}. "
     "Note: the dataset has no crime-rate column, so the Security flag is used as the safety proxy "
     f"(Good-Investment share: secure {df[df.Security_Flag==1].Good_Investment.mean():.0%} vs not {df[df.Security_Flag==0].Good_Investment.mean():.0%}).")

(ROOT / "reports" / "eda_summary.md").write_text("# EDA Findings\n\n" + "\n".join(notes))
print("Saved", len(notes), "charts to", OUT)
