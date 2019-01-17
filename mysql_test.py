import pymysql

db = pymysql.connect("localhost","root","ssc940114","test" )
cursor = db.cursor()
cursor.execute("SELECT VERSION()")
data = cursor.fetchone()
print ("Database version : %s " % data)

cursor.execute("DROP TABLE IF EXISTS employee")
sql = """CREATE TABLE `employee` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `first_name` char(20) NOT NULL,
  `last_name` char(20) DEFAULT NULL,
  `age` int(11) DEFAULT NULL,
  `sex` char(1) DEFAULT NULL,
  `income` float DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;"""
cursor.execute(sql)
print("Created table Successfull.")

sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
   LAST_NAME, AGE, SEX, INCOME)
   VALUES ('Mac', 'Su', 20, 'M', 5000)"""
cursor.execute(sql)
db.commit()

sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
   LAST_NAME, AGE, SEX, INCOME)
   VALUES ('Kobe', 'Bryant', 40, 'M', 8000)"""
cursor.execute(sql)
db.commit()

sql = "select * from employee"
cursor.execute(sql)
results = cursor.fetchall()
for row in results:
    fname = row[1]
    lname = row[2]
    age = row[3]
    sex = row[4]
    income = row[5]
print ("name = %s %s,age = %s,sex = %s,income = %s" % \
             (fname, lname, age, sex, income ))??