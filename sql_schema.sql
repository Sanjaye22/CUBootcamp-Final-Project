- Drop table if exists
Drop Table testfinal;
Drop Table trainfinal;

CREATE TABLE "testfinal" (
    "uniqueID" VARCHAR,
	"drugName" VARCHAR, 
    "condition" VARCHAR,
    "review" VARCHAR,
    "rating" VARCHAR,
	"date" date,
    "usefulCount" VARCHAR
);

CREATE TABLE "trainfinal" (
    "uniqueID" VARCHAR,
	"drugName" VARCHAR, 
    "condition" VARCHAR,
    "review" VARCHAR,
    "rating" VARCHAR,
	"date" date,
    "usefulCount" VARCHAR
);

-- View tables
SELECT * FROM testfinal;
SELECT * FROM trainfinal;
