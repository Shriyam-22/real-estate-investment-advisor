# EDA Findings

**Q1. Distribution of property prices**  
Prices range ₹10–500 L, median ₹254 L; distribution is close to uniform (no strong skew).

**Q2. Distribution of property sizes**  
Sizes range 500–5000 sq ft, median 2747; roughly uniform.

**Q3. Price per sq ft by property type**  
Median ₹/sqft — Apartment: ₹9,245, Independent House: ₹9,305, Villa: ₹9,190. Type barely moves price/sqft.

**Q4. Size vs price**  
Pearson r = -0.003: size and total price are essentially uncorrelated in this dataset.

**Q5. Outliers in price/sqft and size**  
19,723 rows (7.9%) are IQR outliers on price/sqft (small homes with high prices); size has no outliers. Outliers are flagged, not removed.

**Q6. Average price per sq ft by state**  
Highest: Karnataka (₹13,252); lowest: Punjab (₹12,931). Spread is narrow (~2%).

**Q7. Average property price by city**  
Most expensive: Bangalore (₹258.5 L); cheapest: Cuttack (₹250.8 L).

**Q8. Median property age by locality**  
Across 500 localities median age ranges 16–21 yrs; oldest stock: Locality_72, Locality_403, Locality_90.

**Q9. BHK distribution across cities (top 12 by listings)**  
1–5 BHK are evenly represented (~20% each) in every city.

**Q10. Price trend by build year — top 5 most expensive localities**  
Top localities by ₹/sqft: Locality_207, Locality_416, Locality_246, Locality_359, Locality_387. Price vs build year is flat/noisy — no clear vintage premium.

**Q11. Correlation between numeric features**  
Raw features are almost independent of each other (|r|<0.05). Strong links are only engineered ones: Price ↔ Future price, and Price/sqft ↔ Size (inverse).

**Q12. Nearby Schools vs price per sq ft**  
Avg ₹/sqft stays within ₹12,888–13,292 across 1–10 schools: no meaningful effect on current price (but it raises the projected growth rate).

**Q13. Nearby Hospitals vs price per sq ft**  
Avg ₹/sqft stays within ₹12,917–13,257 across 1–10 hospitals: no meaningful effect on current price (but it raises the projected growth rate).

**Q14. Price by furnished status**  
Median price — Furnished: ₹254 L, Semi-furnished: ₹254 L, Unfurnished: ₹254 L.

**Q15. Price per sq ft by facing direction**  
Avg ₹/sqft — East: ₹13,024, North: ₹13,025, South: ₹13,044, West: ₹13,140. Facing has no material effect.

**Q16. Properties by owner type**  
Broker: 83,479, Owner: 83,268, Builder: 83,253.

**Q17. Properties by availability status**  
Under_Construction: 125,035, Ready_to_Move: 124,965.

**Q18. Does parking affect price?**  
Avg price with parking ₹254.7 L vs without ₹254.4 L — negligible difference.

**Q19. Amenities vs price per sq ft and investment quality**  
Amenity count doesn't change ₹/sqft, but Good-Investment share rises from 45% to 80%.

**Q20. Public transport accessibility vs price & investment potential**  
₹/sqft is flat, but Good-Investment share goes Low 37% → High 83% and avg growth 7.5% → 10.1%.

**Q21. Infrastructure score vs Good Investment / resale value**  
Corr(Infrastructure, Future price) = 0.076; Corr(Infrastructure, growth rate) = 0.524. Note: the dataset has no crime-rate column, so the Security flag is used as the safety proxy (Good-Investment share: secure 79% vs not 51%).
