-- IS5102-A2-SQL-220025456
---------------------------------------------------------------------------------------------------
-- Task 3: SQL result
---------------------------------------------------------------------------------------------------

-- 1. List all books published by “Ultimate Books” which are in the “Science Fiction” genre;

|book_id|book_title   |book_author                          |book_publisher|book_genre              |
|-------|-------------|-------------------------------------|--------------|------------------------|
|BK001  |Moon Palace  |Paul Auster                          |Ultimate Books|Science Fiction         |
|BK002  |My Inventions|Nikola Tesla                         |Ultimate Books|Science Fiction         |
|BK003  |Tesla Papers |Nikola Tesla, David Hatcher Childress|Ultimate Books|Science Fiction         |


-- 2. List titles and ratings of all books in the “Science and Technology” genre, 
-- ordered first by rating (top rated first), and then by the title;


|book_title                                             |book_genre              |rating|
|-------------------------------------------------------|------------------------|------|
|Agile Web Development with Rails: A Pragmatic Guide    |Science and Technology  |5     |
|The Zen of CSS Design: Visual Enlightenment for the Web|Science and Technology  |4     |
|HTML  XHTML  and CSS (Visual Quickstart Guide)         |Science and Technology  |1     |


-- 3. List all orders placed by customers with customer address in the city of Edinburgh, 
-- since 2020, in chronological order, latest first;


|order_id|customer_id|customer_name |customer_street     |customer_city|customer_postcode|customer_country|date_ordered|
|--------|-----------|--------------|--------------------|-------------|-----------------|----------------|------------|
|OR003   |CS003      |Ethan Jordan  |Coborn  Way         |Edinburgh    |LT6O-MD8D        |Argentina       |2020-12-08  |
|OR001   |CS001      |Daniel Vince  |Fawn Tunnel         |Edinburgh    |JRYI-PZWR        |Algeria         |2020-11-16  |
|OR013   |CS003      |Ethan Jordan  |Coborn  Way         |Edinburgh    |LT6O-MD8D        |Argentina       |2020-11-14  |
|OR011   |CS001      |Daniel Vince  |Fawn Tunnel         |Edinburgh    |JRYI-PZWR        |Algeria         |2020-10-14  |
|OR006   |CS006      |Peter Keys    |Adderley   Crossroad|Edinburgh    |ED1U-TJR1        |Cambodia        |2020-09-02  |
|OR015   |CS005      |Lily Wilkinson|Oxford Road         |Edinburgh    |PEOQ-CAUU        |Oman            |2020-08-31  |
|OR005   |CS005      |Lily Wilkinson|Oxford Road         |Edinburgh    |PEOQ-CAUU        |Oman            |2020-06-23  |
|OR016   |CS006      |Peter Keys    |Adderley   Crossroad|Edinburgh    |ED1U-TJR1        |Cambodia        |2020-03-03  |



-- 4. List all book editions which have less than 5 items in stock, together with the name, 
-- account number and supply price of the minimum priced supplier for that edition.


|book_id|book_edition|book_qty|supplier_price|supplier_name        |supplier_account|
|-------|------------|--------|--------------|---------------------|----------------|
|BK002  |2,011       |2       |10.1          |Ballantine Books     |7453-42663247   |
|BK008  |2,019       |4       |17.08         |Ballantine Books     |7453-42663247   |
|BK002  |2,011       |2       |10.1          |Random House Trade   |0122-02381004   |
|BK007  |2,016       |3       |6.8           |Vintage International|0782-38041325   |



-- 5. Calculate the total value of all audiobook sales since 2020 for each publisher;


|book_publisher           |SUM(oc.order_qty * book_price)|
|-------------------------|------------------------------|
|Archaia                  |126.6                         |
|Houghton Mifflin Harcourt|50.22                         |
|Shambhala                |142.08                        |
|Ultimate Books           |237.6                         |


-- NEW QUERY 1
-- Which genre is most popular in a country?

|customer_country|book_genre              |max(qty)|
|----------------|------------------------|--------|
|Algeria         |Science Fiction         |4       |
|Argentina       |Science Fiction         |5       |
|Brunei          |Adventure               |2       |
|Cambodia        |Science and Technology  |4       |
|Canada          |Thriller                |16      |
|Cyprus          |Science Fiction         |3       |
|Djibouti        |Adventure               |2       |
|Mozambique      |Science Fiction         |3       |
|Oman            |Science and Technology  |10      |
|Sudan           |Adventure               |2       |


--NEW QUERY 2
-- Get Customer details of highest sales order received by book store;


|order_id|customer_id|customer_name |address                             |cust_phone_number|max(sales)|
|--------|-----------|--------------|------------------------------------|-----------------|----------|
|OR005   |CS005      |Lily Wilkinson|Oxford Road Edinburgh PEOQ-CAUU Oman|+44-1254-850173  |261.15    |



--NEW QUERY 3
-- Get 5 book ids having least margins percentage over supplier price ;


|book_id|book_edition|book_price|supplier_price|margin|
|-------|------------|----------|--------------|------|
|BK005  |2,016       |52.23     |50.23         |3.98  |
|BK008  |2,019       |19.08     |17.8          |7.19  |
|BK001  |2,020       |29.7      |27.7          |7.22  |
|BK009  |2,022       |17.55     |16            |9.69  |
|BK002  |2,011       |11.1      |10.1          |9.9   |


--Result of VIEW 1 and VIEW 2 is presented the report.

