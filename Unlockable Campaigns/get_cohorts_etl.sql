-- ETL table cohorts are calculated on Monday
WITH params AS (
    SELECT '{country_code}'::text AS country_code,
        '{start_date}'::date AS start_date,
        '{city_name}'::text AS city_name
),

/*
rfmv1 AS (
    SELECT 'rfmv1'::text AS method,
        DATE(DATE_TRUNC('week', c.week::date)) AS period,
        c.city_name as city,
        c.country_code as country,
        c.cohort,
        c.user_id
    FROM public.etl_rider_cohorts_weekly_by_city c
    WHERE c.country_code = (SELECT country_code FROM params)
    AND period = DATE(DATE_TRUNC('week', (SELECT start_date FROM params)))
    AND c.city_name = (SELECT city_name FROM params)
)
*/

rfmv2 AS (
    SELECT 'rfmv2'::text AS method,
        DATE(DATE_TRUNC('day', c.week::date)) AS period,
        c.city_name as city,
        c.country_code as country,
        c.cohort,
        c.user_id
    FROM public.etl_rider_cohorts_max_weekly_by_city c
    WHERE c.country_code = (SELECT country_code FROM params)
    AND period = DATE(DATE_TRUNC('day', (SELECT start_date FROM params)))
    AND c.city_name = (SELECT city_name FROM params)
)

/*
SELECT *
FROM rfmv1

UNION ALL
*/

SELECT *
FROM rfmv2


              
              
