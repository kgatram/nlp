/*
 * IS5102-A2-SQL-220025456
 */


/* .mode column
 * .header on
 * .width 18 18 18 18
 */

-- enforce foreign keys check
PRAGMA foreign_keys = TRUE;

------------------------------------------------------
-- TASK 2: DDL 
------------------------------------------------------

DROP TABLE IF EXISTS ord_contains;
DROP TABLE IF EXISTS review;
DROP TABLE IF EXISTS book_order;
DROP TABLE IF EXISTS supply;
DROP TABLE IF EXISTS edition;
DROP TABLE IF EXISTS book;
DROP TABLE IF EXISTS customer_phone;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS supplier_phone;
DROP TABLE IF EXISTS supplier;
DROP TABLE IF EXISTS book_type;
DROP TABLE IF EXISTS genre;



-- Book type table ∈ {audiobook, hardcover, and paperback.} 
CREATE TABLE book_type (
type_id		INTEGER,
description	VARCHAR(50),			
PRIMARY KEY (type_id)
);



CREATE TABLE genre (
genre_id INTEGER,
book_genre 	VARCHAR(50),
primary key (genre_id)
);



CREATE TABLE book (
book_id 		CHAR(5),
book_title 		VARCHAR(50),
book_author 	VARCHAR(50),
genre_id 		INTEGER,
book_publisher 	VARCHAR(50),
PRIMARY KEY (book_id),
FOREIGN KEY (genre_id) REFERENCES GENRE ON DELETE SET NULL );



CREATE TABLE edition (
book_id 		CHAR(5),
book_edition 	INTEGER CHECK(book_edition > 0),
type_id 		INTEGER,
book_price 		NUMERIC(8,2),
book_qty 		INTEGER,
UNIQUE (book_id, book_edition, type_id),
FOREIGN KEY (book_id) REFERENCES book(book_id)
ON UPDATE CASCADE,
FOREIGN KEY (type_id) REFERENCES book_type ON DELETE SET NULL
);



CREATE TABLE customer (
customer_id    		CHAR(5),
customer_name     	VARCHAR(50),
customer_email    	VARCHAR(50),
customer_street   	VARCHAR(50),
customer_city     	VARCHAR(50),
customer_postcode 	CHAR(10),
customer_country  	VARCHAR(50),
PRIMARY KEY (customer_id)
);



--customer phone table 
CREATE TABLE customer_phone (
customer_id			CHAR(5),
cust_phone_type		CHAR(10),
cust_phone_number	CHAR(20),
FOREIGN KEY (customer_id) REFERENCES customer
ON DELETE CASCADE
ON UPDATE CASCADE
);


-- unique index on customer phone
CREATE UNIQUE INDEX customer_phone_uidx ON customer_phone(customer_id, cust_phone_type);


CREATE TABLE review (
customer_id		CHAR(5),
book_id			CHAR(5),
book_edition	INTEGER,
type_id			INTEGER,
rating			INTEGER CHECK(rating BETWEEN 1 AND 5),
UNIQUE (customer_id, book_id, book_edition, type_id),
FOREIGN KEY (customer_id) REFERENCES customer,
FOREIGN KEY (book_id,book_edition,type_id) REFERENCES edition(book_id,book_edition,type_id)
);


-- order table 
CREATE TABLE book_order (
order_id				CHAR(10),
customer_id				CHAR(5),
delivery_street			VARCHAR(50),
delivery_city			VARCHAR(50),
delivery_postcode		CHAR(10),
delivery_country		VARCHAR(50),
date_ordered			DATE,
date_delivered			DATE,
PRIMARY KEY (order_id),
FOREIGN KEY (customer_id) REFERENCES customer
);


-- table containing order details
CREATE TABLE ord_contains (
order_id			CHAR(10),
book_id				CHAR(5),
book_edition		INTEGER,
type_id				INTEGER,		
order_qty			INTEGER,
UNIQUE (order_id, book_id, book_edition, type_id),
FOREIGN KEY (order_id) REFERENCES book_order,
FOREIGN KEY (book_id,book_edition,type_id) REFERENCES edition(book_id,book_edition,type_id)
);


CREATE TABLE supplier (
supplier_id			CHAR(5),
supplier_name		VARCHAR(50),
supplier_account	CHAR(20),
PRIMARY KEY (supplier_id)
);


CREATE TABLE supplier_phone (
supplier_id	    	CHAR(5),
sup_phone_type	    CHAR(10),
sup_phone_number	CHAR(20),
FOREIGN KEY (supplier_id) REFERENCES supplier
ON DELETE CASCADE
ON UPDATE CASCADE
);


-- unique index on supplier phone
CREATE UNIQUE INDEX supplier_phone_uidx ON supplier_phone(supplier_id, sup_phone_type);


CREATE TABLE supply (
supplier_id			CHAR(5),
book_id				CHAR(5),
book_edition		INTEGER,
type_id				INTEGER,
supplier_price		NUMERIC(8,2),
UNIQUE (supplier_id,book_id,book_edition,type_id),
FOREIGN KEY (supplier_id) REFERENCES supplier
ON DELETE CASCADE
ON UPDATE CASCADE,
FOREIGN KEY (book_id,book_edition,type_id) REFERENCES edition(book_id,book_edition,type_id)
);


------------------------------------------------------------------
-- TEST DATA setup 
------------------------------------------------------------------

-- Book type table ∈ {audiobook, hardcover, and paperback.} 
INSERT INTO book_type VALUES
(1, 'audiobook'),
(2, 'hardcover'),
(3, 'paperback');

INSERT INTO genre VALUES 
(1,'Fantasy                 '),
(2,'Adventure               '),
(3,'Romance                 '),
(4,'Contemporary            '),
(5,'Dystopian               '),
(6,'Mystery                 '),
(7,'Horror                  '),
(8,'Thriller                '),
(9,'Paranormal              '),
(10,'Historical fiction      '),
(11,'Science Fiction         '),
(12,'Childrens               '),
(13,'Memoir                  '),
(14,'Cookbook                '),
(15,'Art                     '),
(16,'Self-help               '),
(17,'Development             '),
(18,'Motivational            '),
(19,'Health                  '),
(20,'History                 '),
(21,'Travel                  '),
(22,'Guide / How-to          '),
(23,'Families & Relationships'),
(24,'Humor                   '),
(25,'Science and Technology  ');


INSERT INTO customer VALUES
('CS001','Daniel Vince',   'Daniel_Vince4826@bulaffy.com','Fawn Tunnel',         'Edinburgh', 'JRYI-PZWR','Algeria'),
('CS002','Ron Lloyd',      'Ron_Lloyd8091@twipet.com','Ernest  Street',          'Bucharest', 'VAGV-WAYL','Mozambique'),
('CS003','Ethan Jordan',   'Ethan_Jordan1047@supunk.biz','Coborn  Way',          'Edinburgh', 'LT6O-MD8D','Argentina'),
('CS004','Chanelle Fenton','Chanelle_Fenton1760@cispeto.com','Cam  Alley',       'Norfolk',   'BGLH-7FMX','Cyprus'),
('CS005','Lily Wilkinson', 'Lily_Wilkinson5275@twipet.com','Oxford Road',        'Edinburgh', 'PEOQ-CAUU','Oman'),
('CS006','Peter Keys',     'Peter_Keys4068@bauros.biz','Adderley   Crossroad',   'Edinburgh', 'ED1U-TJR1','Cambodia'),
('CS007','Abbey Harper',   'Abbey_Harper8733@infotech44.tech','Lexington Tunnel','Miami',     'P7UX-YNNV','Canada'),
('CS008','Denny Watson',   'Denny_Watson1991@extex.org','Geary   Hill',          'Glendale',  'IMJJ-E6R8','Djibouti'),
('CS009','Jacqueline Shaw','Jacqueline_Shaw292@bauros.biz','Belgrave  Vale',     'Pittsburgh','JMLM-TEDD','Brunei'),
('CS010','Rihanna Ellis',  'Rihanna_Ellis1658@corti.com','Erindale Walk',        'Seattle',   'SWUB-XGFW','Sudan');


INSERT INTO customer_phone  VALUES
("CS001","mobile",  "+447453426632"),
("CS002","home",    "+44-0122-0238100"),
("CS003","mobile",  "0782380413"),
("CS004","mobile",  "0733257072"),
("CS005","home",    "+44-1254-850173"),
("CS005","mobile",  "073258123"),
("CS006","mobile",  "073258580"),
("CS007","home",    "+22-27251426"),
("CS008","home",    "013254412"),
("CS009","home",    "023810047"),
("CS010","home",    "024772441");




INSERT INTO book VALUES
("BK001","Moon Palace","Paul Auster",	11,"Ultimate Books"),
("BK002","My Inventions","Nikola Tesla",11,"Ultimate Books"),
("BK003","Tesla Papers","Nikola Tesla, David Hatcher Childress",	11,"Ultimate Books"),
("BK004","The Zen of CSS Design: Visual Enlightenment for the Web","Dave Shea,Molly E. Holzschlag",	25,"Houghton Mifflin Harcourt"),
("BK005","Agile Web Development with Rails: A Pragmatic Guide","Dave Thomas,David Heinemeier Hansson,Leon Breedt,Mike Clark,Thomas  Fuchs,Andreas  Schwarz",25,"Pragmatic Bookshelf"),
("BK006","HTML  XHTML  and CSS (Visual Quickstart Guide)","Elizabeth Castro",25,"Atheneum Books for Young Readers: Richard Jackson Books"),
("BK007","The Changeling","Kate Horsley",8,"Shambhala"),
("BK008","The Known World","Edward P. Jones,Kevin R. Free",4,"HarperAudio"),
("BK009","Traders  Guns & Money: Knowns and Unknowns in the Dazzling World of Derivatives","Satyajit Das",17,"FT Press"),
("BK010","Artesia: Adventures in the Known World","Mark Smylie",2,"Archaia");



INSERT INTO edition VALUES
("BK001",2020,1, 29.70,  5),
("BK002",2011,2, 11.10,  2),
("BK003",2018,3, 35.1, 11),
("BK004",2022,1, 25.11, 15),
("BK005",2016,2, 52.23,  8),
("BK006",2021,3, 25.99, 10),
("BK007",2016,1, 8.88,   3),
("BK008",2019,2, 19.08,  4),
("BK009",2022,3, 17.55,  7),
("BK010",2022,1, 21.1, 12);

INSERT INTO review VALUES
("CS001","BK001",2020,1,1 ),
("CS002","BK002",2011,2,2 ),
("CS003","BK003",2018,3,3 ),
("CS004","BK004",2022,1,4 ),
("CS005","BK005",2016,2,5 ),
("CS006","BK006",2021,3,1 ),
("CS007","BK007",2016,1,2 ),
("CS008","BK008",2019,2,3 ),
("CS009","BK009",2022,3,4 ),
("CS010","BK010",2022,1,5 );

INSERT INTO supplier VALUES
("SP001","Ballantine Books",  "7453-42663247"),
("SP002","Random House Trade","0122-02381004"),
("SP003","Vintage International","0782-38041325"),
("SP004","Avon",  "7332-57072725"),
("SP005","Payot","1254-85017325"),
("SP006","Spectra","8501-73258580"),
("SP007","Harper Torch",  "5707-27251426"),
("SP008","Bantam Books","3804-13254412"),
("SP009","VIZ Media LLC","0238-10047181"),
("SP010","Puffin",  "4266-32477244");

INSERT INTO supplier_phone VALUES
("SP001","mobile",  "+447453426632"),
("SP002","work","44-0122-0238100"),
("SP003","business","0782380413"),
("SP004","mobile",    "0733257072"),
("SP005","work","+44-1254-850173"),
("SP006","mobile","073258580"),
("SP007","work",  "+22-27251426"),
("SP008","work","013254412"),
("SP009","work","023810047"),
("SP010","work","024772441");

INSERT INTO supply VALUES
("SP001","BK002",2011,2, 10.10),
("SP001","BK003",2018,3, 30.00),
("SP001","BK004",2022,1, 20.11),
("SP001","BK005",2016,2, 50.23),
("SP001","BK006",2021,3, 21.99),
("SP001","BK001",2020,1, 27.70),
("SP001","BK007",2016,1, 6.88 ),
("SP001","BK008",2019,2, 17.08),
("SP001","BK009",2022,3, 15.55),
("SP001","BK010",2022,1, 19.00),
("SP002","BK001",2020,1, 25.70),
("SP002","BK002",2011,2, 10.10),
("SP002","BK003",2018,3, 25.00),
("SP003","BK006",2021,3, 22),
("SP003","BK007",2016,1, 6.80 ),
("SP003","BK008",2019,2, 17.8),
("SP003","BK009",2022,3, 16.00),
("SP003","BK010",2022,1, 19.10);


-- order 

INSERT INTO book_order VALUES
('OR001','CS001','Chalcot  Way',	'Berna'	,             'OVVL-LMMX',	'Cyprus',	            '2020-11-16',	'2020-02-15'),
('OR002','CS002','Glenwood Hill',	'Las Vegas',	      'WOOR-DRNM',	'Syria',	            '2020-05-26',	'2020-10-04'),
('OR003','CS003','Bliss  Lane',	    'Sacramento',	      'TIP4-JOOZ',	'Estonia',	            '2020-12-08',	'2020-10-29'),
('OR004','CS004','Chandos  Route',	'New Orleans',	      'WDOF-NQ8J',	'Turkmenistan',	        '2020-05-25',	'2020-08-16'),
('OR005','CS005','Cleveland  Avenue',	'Salem'	,         'YJBR-DCMA',	'Senegal',	            '2020-06-23',	'2020-03-09'),
('OR006','CS006','Sheffield Alley',	'Madrid',	          '5BDB-SODK',	'Guinea-Bissau',	    '2020-09-02',	'2020-07-19'),
('OR007','CS007','Becher  Street',	'San Bernardino',     'CVZK-NCCA',	'Qatar',	            '2020-12-29',	'2020-06-06'),
('OR008','CS008','Bellenden   Walk',	'Fort Lauderdale','PONM-R4XI',	'Eritrea',	            '2020-02-01',	'2020-06-06'),
('OR009','CS009','Chandos  Pass',	'Bridgeport',	      'MXED-RVNT',	'Iran',	                '2020-07-30',	'2020-09-12'),
('OR010','CS010','Adams  Lane',	    'Rome',	              'XSUZ-3RWZ',	'Gabon',	            '2020-10-18',	'2020-07-16'),
('OR011','CS001','Camden  Alley',	'Hayward',	          'HLRU-T1H1',	'Trinidad and Tobago',	'2020-10-14',	'2020-02-15'),
('OR012','CS002','Cleveland  Road',	'Oakland',	          'ISNC-ILNR',	'Gabon',	            '2020-01-27',	'2020-03-28'),
('OR013','CS003','Boadicea   Way',	'Oklahoma City',	  '5GVY-QEKA',	'Tunisia',	            '2020-11-14',	'2020-08-07'),
('OR014','CS004','Sundown Avenue',	'Santa Ana',	      'Y50E-U8IP',	'Japan',	            '2020-08-09',	'2020-07-08'),
('OR015','CS005','Kinglake  Road',	'Miami',	          'ZMT9-H6JB',	'Dominica',	            '2020-08-31',	'2020-12-08'),
('OR016','CS006','Timothy  Way',	'Salt Lake City',	  'J5PU-U8ON',	'Israel',	            '2020-03-03',	'2020-04-17'),
('OR017','CS007','Castlereagh   Road',	'Pittsburgh',	  '7LUZ-DUFW',	'Syria',	            '2020-03-26',	'2020-11-23'),
('OR018','CS008','Lonsdale  Drive',	'Baltimore'	,         'ZP6D-MJMT',	'Chad',	                '2020-05-05',	'2020-03-24'),
('OR019','CS009','Yorkshire  Grove',	'Valetta',	      'BVYR-IJLR',	'Suriname',	            '2020-09-07',	'2020-07-13'),
('OR020','CS010','Blandford  Pass',	'Indianapolis',	      '9RJV-H9AH',	'Lebanon',	            '2020-10-21',	'2020-08-16');


-- contains

INSERT INTO ord_contains VALUES
("OR001","BK001",2020,1,2 ),
("OR011","BK001",2020,1,2 ),
("OR012","BK001",2020,1,2 ),
("OR013","BK001",2020,1,2 ),
("OR002","BK002",2011,2,1 ),
("OR020","BK002",2011,2,1 ),
("OR003","BK003",2018,3,3 ),
("OR014","BK003",2018,3,3 ),
("OR004","BK004",2022,1,2 ),
("OR005","BK005",2016,2,5 ),
("OR015","BK005",2016,2,5 ),
("OR006","BK006",2021,3,2 ),
("OR016","BK006",2021,3,2 ),
("OR007","BK007",2016,1,8 ),
("OR017","BK007",2016,1,8 ),
("OR008","BK008",2019,2,1 ),
("OR009","BK009",2022,3,1 ),
("OR010","BK010",2022,1,2 ),
("OR018","BK010",2022,1,2 ),
("OR019","BK010",2022,1,2 );

---------------------------------------------------------------------------------------------------
-- Task 3: SQL
---------------------------------------------------------------------------------------------------

-- 1. List all books published by “Ultimate Books” which are in the “Science Fiction” genre;

SELECT
	book_id ,
	book_title ,
	book_author,
	book_publisher ,
	g.book_genre
FROM
	book b
NATURAL JOIN genre g
WHERE
	LOWER(book_publisher) = 'ultimate books'
	AND genre_id = 11;


-- 2. List titles and ratings of all books in the “Science and Technology” genre, 
-- ordered first by rating (top rated first), and then by the title;

SELECT
	book_title ,
	book_genre,
	rating
FROM
	book b
NATURAL JOIN genre g
NATURAL JOIN review r
WHERE
	genre_id = 25
ORDER BY
	rating DESC ,
	book_title ;


-- 3. List all orders placed by customers with customer address in the city of Edinburgh, 
-- since 2020, in chronological order, latest first;

SELECT
	order_id ,
	customer_id ,
	c.customer_name ,
	c.customer_street ,
	c.customer_city ,
	c.customer_postcode,
	c.customer_country ,
	date_ordered
FROM
	book_order b
NATURAL JOIN customer c
WHERE
	LOWER(customer_city) = 'edinburgh'
	AND date_ordered >= DATE('2020-01-01') 
ORDER BY
	date_ordered DESC ; 


-- 4. List all book editions which have less than 5 items in stock, together with the name, 
-- account number and supply price of the minimum priced supplier for that edition.

SELECT
	book_id ,
	book_edition,
	book_qty,				
	supplier_price ,		
	supplier_name,			
	supplier_account		
FROM
	supply s
NATURAL JOIN edition e
NATURAL JOIN supplier s2
WHERE
	book_qty < 5
	AND supplier_price IN (
	SELECT
		MIN(supplier_price)
	FROM
		supply s3
	GROUP BY
		book_id,
		book_edition) ;
	

-- 5. Calculate the total value of all audiobook sales since 2020 for each publisher;
	
SELECT
	b2.book_publisher ,						
	SUM(oc.order_qty * book_price)			
FROM
	ord_contains oc
NATURAL JOIN book_order b
NATURAL JOIN edition b1
NATURAL JOIN book b2
WHERE
	type_id = 1
	AND b.date_delivered >= DATE('2020-01-01')
GROUP BY 
	b2.book_publisher ;


-- NEW QUERY 1
-- Which genre is most popular in a country?


with popular AS  (
SELECT
	c.customer_country,	
	book_genre,
sum(oc.order_qty) as qty
FROM
	book b
NATURAL JOIN genre g
NATURAL JOIN ord_contains oc
NATURAL JOIN book_order bo
NATURAL JOIN customer c
GROUP BY 
c.customer_country,
book_genre)
SELECT customer_country,	
	book_genre, max(qty) FROM popular
	GROUP BY 
customer_country;


--NEW QUERY 2
-- Get Customer details of highest sales order received by book store;

with sales_order as (
SELECT
	order_id,
	order_qty,
	book_price,
	(order_qty * book_price) as sales,
	customer_id,
	customer_name,
	customer_email,
	(customer_street || ' ' || customer_city || ' ' || customer_postcode || ' ' || customer_country) as address,
	cust_phone_number
FROM
	ord_contains oc
NATURAL JOIN edition e
NATURAL JOIN book_order bo
NATURAL JOIN customer c
NATURAL JOIN customer_phone )
select
	order_id ,
	customer_id,
	customer_name,
	address,
	cust_phone_number,
	max(sales)
from
	sales_order;


--NEW QUERY 3
-- Get 5 book ids having least margins percentage over supplier price ;


SELECT
	book_id ,
	book_edition,
	book_price ,
	s.supplier_price ,
	round(((book_price - s.supplier_price) / s.supplier_price) * 100,
	2) AS margin
FROM
	edition e
NATURAL JOIN supply s 
order by margin ASC 
LIMIT  5 ;


--VIEW 1

DROP VIEW IF EXISTS book_view ; 
CREATE VIEW book_view as 
SELECT
	book_id,
	book_edition ,
	book_title ,
	book_author ,
	book_price,
	book_publisher,
	description,
	book_genre
FROM
	book b
NATURAL JOIN edition e
NATURAL JOIN book_type bt
NATURAL JOIN genre g ;

SELECT * FROM book_view ;


--VIEW 2

DROP VIEW IF EXISTS order_view ;
CREATE VIEW order_view as
SELECT
	order_id,
	book_title,
	book_edition,
	book_price,
	order_qty,
	customer_name,
	(customer_street || ' ' || customer_city || ' ' || customer_postcode || ' ' || customer_country) as address
FROM
	ord_contains oc
natural join edition e
natural join book
natural join book_order bo
natural join customer c ; 

SELECT * FROM order_view;








