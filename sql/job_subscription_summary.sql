-- Segment subscription crosstab by job (counts of no/yes).
SELECT
    job,
    SUM(CASE WHEN y = 'no' THEN 1 ELSE 0 END)  AS "no",
    SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS "yes"
FROM campaign
GROUP BY job
ORDER BY job;
