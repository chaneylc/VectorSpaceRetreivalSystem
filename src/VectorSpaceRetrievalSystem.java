import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class VectorSpaceRetrievalSystem {

    public static void main(String[] args) throws Exception {

        HashMap<String, String> posting = new HashMap<String, String>();
        HashMap<Integer, Double> docLengths = new HashMap<Integer, Double>();
        HashMap<Integer, ArrayList<Double>> docWeights = new HashMap<Integer, ArrayList<Double>>();

        String query = args[0];
        if(query == null) {
            System.err.println("No query string defined.");
            System.exit(1);
        }
        //else
        List<String> rawPosting = Files.readAllLines(Paths.get(args[1]));
        List<String> documentLengths = Files.readAllLines(Paths.get(args[2]));
        List<String> stopWords = Files.readAllLines(Paths.get(args[3]));

        /**
         * read document weight vector and vector length from file
         */
        for(String s : documentLengths) {
            StringTokenizer st = new StringTokenizer(s, "\t ");
            Integer docid = Integer.parseInt(st.nextToken());
            Double length = Double.parseDouble(st.nextToken());
            ArrayList<Double> v = new ArrayList<Double>();
            while(st.hasMoreTokens())
                v.add(Double.parseDouble(st.nextToken()));
            docWeights.put(docid, v);
            docLengths.put(docid, length);
        }

        /**
         * read posting input from file
         */
        for(String s : rawPosting) {
            StringTokenizer st = new StringTokenizer(s, "\t");
            String term = st.nextToken();
            posting.put(term, st.nextToken());
        }

        /**
         * Split the input query, stem the input, count # of tokens
         * calculate query magnitude and vector values: f/max(f) * idf
         */
        HashMap<Integer, Double> cosSims = new HashMap<Integer, Double>();
        String[] terms = stemQuery(query.split(" "), stopWords);
        Integer N = docLengths.size();
        ArrayList<Double> queryVector = new ArrayList<Double>();
        Double queryLength = 0.0;
        for(String s : terms) {
            Double occurrences = 0.0;
            HashMap<String, Double> uniqTerms = new HashMap<String, Double>();
            for(String t : terms) {
                if(uniqTerms.containsKey(t)) {
                    uniqTerms.put(t, uniqTerms.get(t) + 1);
                } else {
                    uniqTerms.put(t, 1.0);
                }
                if(t.equals(s)) occurrences++;
            }
            Double qmaxF = 0.0;
            for(String t : uniqTerms.keySet()) {
                if(uniqTerms.get(t) > qmaxF) qmaxF = uniqTerms.get(t);
            }
            if(posting.containsKey(s)) {
                String p = posting.get(s);
                Integer qdf = calculateDf(p);
                Double qidf = (Math.log10(N / qdf) / Math.log10(2));
                Double w = (occurrences / qmaxF) * qidf;
                StringTokenizer st = new StringTokenizer(p, "\t ");
                Integer docdf = Integer.parseInt(st.nextToken());
                while(st.hasMoreTokens()) {
                    Integer docid = Integer.parseInt(st.nextToken());
                    Integer maxF = Integer.parseInt(st.nextToken());
                    Integer f = Integer.parseInt(st.nextToken());
                    Double idf = (Math.log10(N / docdf) / Math.log10(2));
                    Double tf = f.doubleValue() / maxF.doubleValue();
                    if(cosSims.containsKey(docid)) {
                        Double currentVal = cosSims.get(docid);
                        cosSims.put(docid, currentVal + w * tf * idf);
                    } else {
                        cosSims.put(docid, w * tf * idf);
                    }
                }
                queryLength += Math.pow(w, 2);
            }
        }

        for(Integer docid : cosSims.keySet()) {
            Double currentVal = cosSims.get(docid);
            cosSims.put(docid, currentVal / Math.sqrt(queryLength * docLengths.get(docid)));
        }

        Double[] values = cosSims.values().toArray(new Double[]{});
        Arrays.sort(values);
        for(int i = values.length - 1; i > values.length - 50; i = i - 1) {
            for(Integer docid : cosSims.keySet()) {
                if(cosSims.get(docid) == values[i]) {
                    System.out.println(docid + " : " + values[i]);
                }
            }
        }
    }

    public static String[] stemQuery(String[] terms, List<String> stopWords) {
        ArrayList<String> result = new ArrayList<String>();
        for(String s : terms) {
            Stemmer stem = new Stemmer();
            stem.add(s.toCharArray(), s.length());
            stem.stem();
            String r = stem.toString();
            if(r.length() > 2 && !stopWords.contains(r)) result.add(r);
        }
        return result.toArray(new String[]{});
    }

    public static Integer calculateDf(String s) {
        StringTokenizer st = new StringTokenizer(s, " ");
        return Integer.parseInt(st.nextToken()); // get saved df
    }
}