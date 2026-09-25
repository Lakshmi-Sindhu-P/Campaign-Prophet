-- Segment expected value by job, parameterized by economics scenario.
-- Portable SQL (SQLite dialect); :cost and :revenue are named parameters.
SELECT
    job,
    COUNT(*)                                                              AS total_contacts,
    SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END)                            AS successful_contacts,
    AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END)                        AS success_rate,
    AVG(duration)                                                         AS avg_duration,
    AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END) * :revenue - :cost     AS expected_value_per_contact,
    COUNT(*) * :cost                                                      AS total_cost,
    SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) * :revenue                 AS total_revenue,
    SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) * :revenue - COUNT(*) * :cost AS total_profit,
    100.0 * (SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) * :revenue - COUNT(*) * :cost)
        / (COUNT(*) * :cost)                                              AS roi_percent
FROM campaign
GROUP BY job
ORDER BY expected_value_per_contact DESC;
