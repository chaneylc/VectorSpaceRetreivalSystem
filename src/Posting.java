import javafx.util.Pair;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by chaneylc on 9/26/15.
 */
public class Posting {
    /** Data structure to encode index posting s.a
     * term1 -> {docFrequency = 2, [(docId=1, termFreq=3), (docId=2, termFreq=1)]}
     * term2 -> ...
     * ...
     * termn**/
    private Integer N;
    private HashMap<String, Double> inverseDocumentFrequencies;
    public HashMap<Integer, Double> weights;
    public HashMap<String, Pair<Integer, HashMap<Integer, Integer>>> posting;
    private HashMap<Integer, Integer> maxFrequencies;

    public Posting() {
        this.maxFrequencies = new HashMap<Integer, Integer>();
        this.posting = new HashMap<String, Pair<Integer, HashMap<Integer, Integer>>>();
    }

    public Double getIDF(String term) { return this.inverseDocumentFrequencies.get(term); }
    public Double getWeight(Integer docid) { return this.weights.get(docid); }
    public Integer getFrequency(Integer docid) { return this.maxFrequencies.get(docid); }

    private void calculateIDFs() {
        this.N = this.maxFrequencies.size();
        this.inverseDocumentFrequencies = new HashMap<String, Double>();
        for(String term : posting.keySet()) {
            Integer df = this.posting.get(term).getKey();
            this.inverseDocumentFrequencies.put(term, Math.log(df) / Math.log(2));
        }
    }

    public void calculateWeights() {
        this.calculateIDFs();
        this.weights = new HashMap<Integer, Double>();
        for(String term : posting.keySet()) {
            HashMap<Integer, Integer> docFreqs = this.posting.get(term).getValue();
            for(Integer docid : docFreqs.keySet()) {
                Integer tf = docFreqs.get(docid);
                Double currentWeight = this.weights.get(docid);
                Double idf = this.inverseDocumentFrequencies.get(term);
                this.weights.put(docid, currentWeight + Math.pow(tf * idf, 2));
            }
        }
        for(Integer docid : this.weights.keySet())
            this.weights.put(docid, Math.sqrt(this.weights.get(docid)));
    }

    public void post(String term, Integer docid) {
        if(this.posting.containsKey(term)) { //check if currrent term is in posting
            if (this.posting.get(term).getValue().containsKey(docid)) { //check there is a term freq for current document
                int oldVal = this.posting.get(term).getValue().get(docid);
                this.posting.get(term).getValue().put(docid, oldVal + 1);
                if(this.maxFrequencies.containsKey(docid))
                    if(this.maxFrequencies.get(docid) < oldVal + 1)
                        this.maxFrequencies.put(docid, oldVal + 1);
                else
                    this.maxFrequencies.put(docid, 1);

            } else { //increment document freqeuncy and add key/val pair <docid, 1>
                Pair<Integer, HashMap<Integer, Integer>> old = this.posting.get(term);
                Integer numDocs = old.getKey() + 1;
                HashMap<Integer, Integer> newMap = this.posting.get(term).getValue();
                newMap.put(docid, 1);
                this.posting.put(term, new Pair<Integer, HashMap<Integer, Integer>>(numDocs, newMap));
            }
        } else {
            HashMap<Integer, Integer> newMap = new HashMap<Integer, Integer>();
            newMap.put(docid, 1);
            this.posting.put(term, new Pair<Integer, HashMap<Integer, Integer>>(1, newMap));
            if(!this.maxFrequencies.containsKey(docid)) {
                this.maxFrequencies.put(docid, 1);
            }
        }
    }

    public void writeToFile(String output) throws IOException {
        PrintWriter pw = new PrintWriter(output);
        StringBuilder sb = new StringBuilder();
        for(String term : this.posting.keySet()) {
            sb.append(term + " " + this.posting.get(term).getKey());
            final HashMap<Integer, Integer> documentFrequencies = this.posting.get(term).getValue();
            for(Integer docid : documentFrequencies.keySet()) {
                sb.append(" " + docid + " " + documentFrequencies.get(docid));
            }
            sb.append("\n");
        }
        pw.println(sb.toString());
    }

    public String toString() {

        StringBuilder sb = new StringBuilder();

        sb.append("Total number of documents: " + this.maxFrequencies.size() + "\n");

        for(Integer key : this.maxFrequencies.keySet())
            sb.append("Max Frequency for document: " + key + " = " + this.maxFrequencies.get(key) + "\n");

        /*for(String term : this.posting.keySet()) {
            sb.append("tdf{" + term + "} = " + this.posting.get(term).getKey());
            final HashMap<Integer, Integer> documentFrequencies = this.posting.get(term).getValue();
            for(Integer docid : documentFrequencies.keySet()) {
                    sb.append("\n\t"+term+" is in document: " + docid + " " + documentFrequencies.get(docid) + " times.");
            }
            sb.append("\n");
        }*/
        return sb.toString();
    }
}
