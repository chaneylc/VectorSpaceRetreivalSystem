import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.StringTokenizer;
import java.io.IOException;
import java.util.HashMap;

import javafx.util.Pair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * Created by chaneylc on 9/25/15.
 */
public class InputPreprocessor {

    private HashSet<Integer> N;

    public static class SGMLTokenizerMapper extends Mapper<Object, Text, Text, Text>{

        private Posting P;

        protected void setup(Context context) {
            this.P = new Posting();
        }

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();

            //replace all SGML tag components, maybe a seperate map reduce job would make this more efficient.
            StringTokenizer getId = new StringTokenizer(value.toString().replaceAll("(\\<.*?\\>)", ""), " ");
            getId.nextToken(); //kill cranfield
            //output for this map will emit (docid, List<String>) where the Strings have no SGML components but have not been tokenized
            int docid = Integer.parseInt(getId.nextToken());

            StringBuilder noCranfieldNoDocId = new StringBuilder();
            while (getId.hasMoreTokens()) noCranfieldNoDocId.append(getId.nextToken() + " ");

            StringTokenizer st = new StringTokenizer(noCranfieldNoDocId.toString(), " :\t\n0123456789.,\\/'~`!?!@#$%^&*()_+-=");

            while (st.hasMoreElements()) {
                String next = st.nextToken().toLowerCase().trim();
                /**
                 * I noticed that some of the stemmed output still had words like: 'ae', 'ce', 'z'
                 * so I limit the words to more than 2 characters
                 */
                if ((conf.get(next) == null) && next.length() > 2) {
                    /**
                     * naively use new Stemmer for each word,
                     * probably unnecessary and very inefficient for map reduce jobs
                     * Maybe there is a distributed implementation of stemming?
                     */
                    Stemmer stemmer = new Stemmer(); //Porter stemmer implementation
                    stemmer.add(next.toCharArray(), next.length());
                    stemmer.stem();
                    String stemmedWord = stemmer.toString();
                    this.P.post(stemmedWord, docid);
                    //context.write(new IntWritable(docid), new Text(stemmedWord));
                }
            }
        }

        protected void cleanup(Context context) throws IOException, InterruptedException {
            //System.out.println(this.posting.toString());
            for(String term : this.P.posting.keySet()) {
                HashMap<Integer, Integer> docFreqs = this.P.posting.get(term).getValue();
                for(Integer docid : docFreqs.keySet())
                    context.write(new Text(term), new Text(docid + " " + docFreqs.get(docid).toString()));
            }
        }
    }

    public static class RawInputReducer extends Reducer<Text,Text,Text,Text> {

        private HashSet<Integer> N;
        private HashMap<Integer, Integer> maxFrequencies;
        private HashMap<String, Pair<Integer, Text>> encodedPosting;

        protected void setup(Context ctx) {
            this.N = new HashSet<Integer>();
            this.maxFrequencies = new HashMap<Integer, Integer>();
            this.encodedPosting = new HashMap<String, Pair<Integer, Text>>();
        }

        public void reduce(Text key, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {

            Configuration conf = ctx.getConfiguration();
            StringBuilder sb = new StringBuilder();
            int localDocCount = 0;
            for(Text term : values) {
                localDocCount++;
                StringTokenizer st = new StringTokenizer(term.toString(), " ");
                Integer docid = Integer.parseInt(st.nextToken());
                Integer f = Integer.parseInt(st.nextToken());
                if(this.maxFrequencies.containsKey(docid)) {
                    if(this.maxFrequencies.get(docid) < f)
                        this.maxFrequencies.put(docid, f);
                } else {
                    this.maxFrequencies.put(docid, f);
                }
                N.add(docid);
                sb.append(' ');
                sb.append(term);
            }
            this.encodedPosting.put(key.toString(), new Pair<Integer, Text>(localDocCount, new Text(sb.toString())));
        }

        protected void cleanup(Context ctx) throws IOException, InterruptedException {
            ctx.getCounter("N", "N").increment(N.size());
            for(String term : this.encodedPosting.keySet()) {
                Pair<Integer, Text> post = this.encodedPosting.get(term);
                StringTokenizer st = new StringTokenizer(post.getValue().toString(), " ");
                StringBuilder sb = new StringBuilder();
                while(st.hasMoreTokens()) {
                    Integer docid = Integer.parseInt(st.nextToken());
                    Integer maxF = this.maxFrequencies.get(docid);
                    String f = st.nextToken();
                    sb.append(docid);
                    sb.append(' ');
                    sb.append(maxF);
                    sb.append(' ');
                    sb.append(f);
                    sb.append(' ');
                }
                ctx.write(new Text(term), new Text(post.getKey() + " " + sb.toString()));
            }
        }
    }

    public static class PostingTokenizerMapper extends Mapper<Object, Text, IntWritable, DoubleWritable> {

        private Long N;

        protected void setup(Context ctx) {
            this.N = Long.parseLong(ctx.getConfiguration().get("N"));
        }

        public void map(Object key, Text value, Context ctx) throws IOException, InterruptedException {
            /**
             * Posting is in the form of key:term value:docCount docid1 maxFreqDoc1 tf1 ... docidn maxFreqDocN tfn
             */
            StringTokenizer st = new StringTokenizer(value.toString(), "\t "); //key val separated by tab, vals by space
            String term = st.nextToken(); //kill the term
            Integer docCount = Integer.parseInt(st.nextToken());
            while(st.hasMoreTokens()) {
                Integer docid = Integer.parseInt(st.nextToken());
                Double maxF = Double.parseDouble(st.nextToken());
                Double f = Double.parseDouble(st.nextToken());
                Double idf = Math.log10(this.N.doubleValue() / docCount.doubleValue()) / Math.log10(2.0);
                Double tf = f / maxF;
                ctx.write(new IntWritable(docid), new DoubleWritable(tf * idf));
            }
        }
    }

    public static class WeightAggregatorReducer extends Reducer<IntWritable,DoubleWritable,IntWritable,Text> {

        public void reduce(IntWritable key, Iterable<DoubleWritable> tfs, Context ctx) throws IOException, InterruptedException {
            Double sum = 0.0;
            StringBuilder sb = new StringBuilder(); //create vector for document and length
            for(DoubleWritable w : tfs) {
                sb.append(w.get());
                sb.append(' ');
                sum += Math.pow(w.get(), 2); //increment weight by (f * idf)^2
            }
            ctx.write(key, new Text(sum.toString() + "\t" + sb.toString()));
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        /** check for any path errors **/
        Path stopWords = new Path(args[0]);
        Path preInput = new Path(args[1]);
        Path preOutput = new Path(args[2]);
        Path postOutput = new Path(args[3]);
        /** delete output folder if it alrady exists **/

        if(!fs.exists(preInput) || !fs.exists(stopWords))
            System.err.println("Error reading input paths" + preInput + " " + stopWords);
        if(fs.exists(preOutput))
            fs.delete(preOutput, true); //delete output folder before next Hadoop instance runs
        if(fs.exists(postOutput))
            fs.delete(postOutput, true);

        /**
         * pre-pre processing ;)
         * We need some way to access stop words inside the mapper class, in Hadoop 2.0
         * we can specify these values with conf.set (which is what I do) but I'm not sure if this is inefficient.
         */
        for(String stopWord: Files.readAllLines(Paths.get(args[0])))
            conf.set(stopWord, "1"); //set value 'stop word' to 1 if it is a stop word


        Job preprocessJob = Job.getInstance(conf, "SGML Raw Input Preprocessor");
        preprocessJob.setJarByClass(VectorSpaceRetrievalSystem.class);
        preprocessJob.setMapperClass(SGMLTokenizerMapper.class);
        preprocessJob.setReducerClass(RawInputReducer.class);
        preprocessJob.setOutputKeyClass(Text.class);
        preprocessJob.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(preprocessJob, preInput);
        FileOutputFormat.setOutputPath(preprocessJob, preOutput);

        int preResult = preprocessJob.waitForCompletion(true) ? 0 : 1;

        if(preResult == 1) {
            System.err.println("Something went wrong in the preprocessing MapReduce job.");
            System.exit(preResult);
        }

        Long n = preprocessJob.getCounters().findCounter("N", "N").getValue();
        conf.set("N", n.toString());

        Job weightJob = Job.getInstance(conf, "Aggregate Weights");
        weightJob.setJarByClass(VectorSpaceRetrievalSystem.class);
        weightJob.setMapperClass(PostingTokenizerMapper.class);
        weightJob.setReducerClass(WeightAggregatorReducer.class);
        weightJob.setOutputKeyClass(IntWritable.class);
        weightJob.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.addInputPath(weightJob, preOutput);
        FileOutputFormat.setOutputPath(weightJob, postOutput);

        int postResult = weightJob.waitForCompletion(true) ? 0 : 1;

        System.exit(postResult);
    }
}

