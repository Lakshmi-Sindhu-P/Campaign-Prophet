-- Segment subscription crosstab by previous-outcome.
SELECT
    poutcome,
    SUM(CASE WHEN y = 'no' THEN 1 ELSE 0 END)  AS "no",
    SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS "yes"
FROM campaign
GROUP BY poutcome
ORDER BY poutcome;
