USE NOME DO SEU SCHEMA AQUI;

DROP TABLE IF EXISTS registros_biodiversidade;

CREATE TABLE registros_biodiversidade (
    id INT AUTO_INCREMENT PRIMARY KEY,
    barcode TEXT,
    collectioncode TEXT,
    catalognumber TEXT,
    scientificname TEXT,
    kingdom TEXT,
    family TEXT,
    genus TEXT,
    yearcollected TEXT,
    monthcollected TEXT,
    daycollected TEXT,
    country TEXT,
    stateprovince TEXT,
    county TEXT,
    locality TEXT,
    institutioncode TEXT,
    phylum TEXT,
    basisofrecord TEXT,
    verbatimlatitude TEXT,
    verbatimlongitude TEXT,
    identifiedby TEXT,
    collectionid INT,
    specificepithet TEXT,
    recordedby TEXT,
    decimallongitude TEXT,
    decimallatitude TEXT,
    modified TEXT,
    scientificnameauthorship TEXT,
    recordnumber TEXT,
	occurrenceremarks TEXT
);

SET SQL_SAFE_UPDATES = 0; # caso ele diga que está no safe mode

SELECT COUNT(*) FROM registros_biodiversidade;
SELECT * FROM registros_biodiversidade;
