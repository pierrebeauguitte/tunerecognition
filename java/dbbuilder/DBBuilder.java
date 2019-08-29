import org.json.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.StringBuilder;
import java.sql.*;
import abc.parser.TuneParser;
import abc.parser.AbcTune;

public class DBBuilder {

    static final String TABLENAME = "Tunes";

    public static String readFile(String filename) {
	System.out.println("Reading from file");
	String result = "";
	try {
	    BufferedReader br = new BufferedReader(new FileReader(filename));
	    StringBuilder sb = new StringBuilder();
	    String line;
	    while ((line = br.readLine()) != null)
		sb.append(line);
	    br.close();
	    result = sb.toString();
	} catch(Exception e) { e.printStackTrace(); }
	return result;
    }

    public static Connection connect() {
	Connection c = null;
	try {
	    Class.forName("org.sqlite.JDBC");
	    c = DriverManager.getConnection("jdbc:sqlite:corpus.db");
	    c.setAutoCommit(false);
	    System.out.println("Opened database successfully");
	    String checkTable = "SELECT name FROM sqlite_master " +
		"WHERE type='table' AND name='" + TABLENAME + "'";
	    Statement stmt = c.createStatement();
	    ResultSet rs = stmt.executeQuery(checkTable);
	    if (rs.next())
		System.out.println("Table already exists");
	    else {
		System.out.println("Needs to create table");
		String create = "CREATE TABLE " + TABLENAME + "(" +
		    "ID INT NOT NULL, " +
		    "SETTING INT NOT NULL, " +
		    "NAME TEXT, " +
		    "TYPE CHAR(50), " +
		    "MODE CHAR(10), " +
		    "METER CHAR(10), " +
		    "ABC TEXT, " +
		    "KEY TEXT, " +
		    "PARSED TINYINT, " +
		    "PCHIST TEXT, " +
		    "PARSED2 TINYINT," +
		    "PRIMARY KEY (ID, SETTING))";
		stmt.executeUpdate(create);
		System.out.println("Table created successfully");
	    }
	    rs.close();
	    stmt.close();
	    return c;
	} catch (Exception e) {
	    e.printStackTrace();
	    return null;
	}
    }

    public static void populate(JSONArray tunes, Connection con) {
	String insert = "INSERT OR REPLACE INTO Tunes VALUES (?,?,?,?,?,?,?,?,?,?,?)";
	TuneParser tp = new TuneParser();
	try (PreparedStatement ps = con.prepareStatement(insert)) {
	    JSONObject tune;
	    int N = tunes.length();
	    for (int i=0; i<N; i++) {
		tune = tunes.getJSONObject(i);
		System.out.println("Tune " + tune.getInt("tune") +
				   "/" + tune.getInt("setting"));
		String mode = tune.getString("mode");
		String meter = tune.getString("meter"); 
		String abc = tune.getString("abc").replace("\\\\", "");
		String strHist = "";
		try {
		    AbcTune tunetext = tp.parse("K:" + mode + "\n" +
						"M:" + meter + "\n" +
						abc + "\n");
		    long[] hist = ABCSpeller.toPitchClassHistogram(tunetext);
		    JSONArray jsHist = new JSONArray(hist);
		    strHist = jsHist.toString();		    
		} catch (Exception e) {
		    System.out.println("Failed on tune " + tune.getInt("setting"));
		    System.out.println(abc.replace("\r", ""));
		    e.printStackTrace();
		}
		ps.setInt(1,     tune.getInt("tune"));
		ps.setInt(2,     tune.getInt("setting"));
		ps.setString(3,  tune.getString("name"));
		ps.setString(4,  tune.getString("type"));
		ps.setString(5,  mode);
		ps.setString(6,  meter);
		ps.setString(7,  abc.replace("\r", ""));
		String searchKey = MattABCTools.stripAll(tune.getString("abc"));
		int lKey = searchKey.length();
		if (lKey > 2)
		    searchKey += searchKey.substring(0, lKey/2);
		ps.setString(8,  searchKey);
		ps.setInt(9,     MattABCTools.flag ? 1 : 0);
		ps.setString(10, strHist);
		ps.setInt(11,    tp.flag ? 1 : 0);
		ps.executeUpdate();
	    }
	    con.commit();
	} catch (SQLException e) {
	    e.printStackTrace();
	} 
    }

    public static void main (String[] args) {
	Connection c = connect();
	
	String dump = readFile("tunes.json");
	JSONArray arr = new JSONArray(dump);
	populate(arr, c);
    
	try {
	    c.close();
	} catch (Exception e) { e.printStackTrace(); }
    }
}
