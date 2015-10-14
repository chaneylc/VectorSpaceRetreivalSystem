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

    public static class TokensMapper extends Mapper<Object,Text,IntWritable,Text> {

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();

            //replace all SGML tag components, maybe a separate MR job specific for REs would make this more efficient.
            StringTokenizer st = new StringTokenizer(value.toString().replaceAll("(\\<.*?\\>)", ""), " ");
            st.nextToken(); //kill cranfield

            int docid = Integer.parseInt(st.nextToken());

            StringBuilder noCranfieldNoDocId = new StringBuilder();
            while (st.hasMoreTokens()) noCranfieldNoDocId.append(st.nextToken() + " ");

            st = new StringTokenizer(noCranfieldNoDocId.toString(), " :\t\n0123456789.,\\/'~`!?!@#$%^&*()_+-=");

            Stemmer stemmer = new Stemmer(); //Porter stemmer implementation

            while (st.hasMoreElements()) {
                String next = st.nextToken().toLowerCase().trim();
                /**
                 * I noticed that some of the stemmed output still had words like: 'ae', 'ce', 'z'
                 * so I limit the words to more than 2 characters
                 */
                if ((conf.get(next) == null) && next.length() > 2) {
                    /**
                     * calling Porter may be inefficient for map reduce jobs;
                     * The implementation seems very efficient but,
                     * maybe there is a distributed implementation of stemming.
                     */
                    stemmer.add(next.toCharArray(), next.length());
                    stemmer.stem();

                    context.write(new IntWritable(docid), new Text(stemmer.toString()));
                }
            }
        }
    }

    public static class PostingReducer extends Reducer<IntWritable,Text,Text,Text> {

        private Posting P;
        private HashSet<IntWritable> N;

        protected void setup(Context ctx) {
            this.P = new Posting();
            this.N = new HashSet<IntWritable>();
        }

        public void reduce(IntWritable docid, Iterable<Text> stemmedWords, Context ctx) throws IOException, InterruptedException {
            for(Text term : stemmedWords)
                this.P.post(term.toString(), docid.get());
            this.N.add(docid);
        }

        /**
         * Cleanup will emit terms as Text keys and posting lists as Text values
         */
        protected void cleanup(Context context) throws IOException, InterruptedException {
            context.getCounter("N", "N").increment(N.size());
            for(String term : this.P.posting.keySet()) {
                StringBuilder sb = new StringBuilder();
                final HashMap<Integer, Integer> docFreqs = this.P.posting.get(term).getValue();
                for(Integer docid : docFreqs.keySet())
                    sb.append(docid + " " + docFreqs.get(docid).toString() + " ");
                context.write(new Text(term), new Text(sb.toString()));
            }
        }
    }

    public static class IdentityMapper extends Mapper<Object,Text,Text,Text> {
        public void map(Object key, Text value, Context ctx) throws IOException, InterruptedException {
            StringTokenizer st = new StringTokenizer(value.toString(), "\t");
            ctx.write(new Text(st.nextToken()), new Text(st.nextToken()));
        }
    }

    public static class FrequencyReducer extends Reducer<Text,Text,Text,Text> {

        private HashMap<Integer, Integer> maxFrequencies;
        private HashMap<String, Pair<Integer, Text>> encodedPosting;

        protected void setup(Context ctx) {
            this.maxFrequencies = new HashMap<Integer, Integer>();
            this.encodedPosting = new HashMap<String, Pair<Integer, Text>>();
        }

        public void reduce(Text key, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {

            int localDocCount = 0;
            String list = values.iterator().next().toString();
            StringTokenizer st = new StringTokenizer(list, " ");
            while(st.hasMoreTokens()) {
                localDocCount++;
                Integer docid = Integer.parseInt(st.nextToken());
                Integer f = Integer.parseInt(st.nextToken());
                if(this.maxFrequencies.containsKey(docid)) {
                    if(this.maxFrequencies.get(docid) < f)
                        this.maxFrequencies.put(docid, f);
                } else {
                    this.maxFrequencies.put(docid, f);
                }
            }
            this.encodedPosting.put(key.toString(), new Pair<Integer, Text>(localDocCount, new Text(list)));
        }

        protected void cleanup(Context ctx) throws IOException, InterruptedException {
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

    public static class WeightMapper extends Mapper<Object,Text,IntWritable,DoubleWritable> {

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

    public static class WeightSumReducer extends Reducer<IntWritable,DoubleWritable,IntWritable,Text> {

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
        Path freqOutput = new Path(args[4]);
        /** delete output folder if it alrady exists **/

        if(!fs.exists(preInput) || !fs.exists(stopWords))
            System.err.println("Error reading input paths" + preInput + " " + stopWords);
        if(fs.exists(preOutput))
            fs.delete(preOutput, true); //delete output folder before next Hadoop instance runs
        if(fs.exists(postOutput))
            fs.delete(postOutput, true);
        if(fs.exists(freqOutput))
            fs.delete(freqOutput, true);

        /**
         * pre-pre processing ;)
         * We need some way to access stop words inside the mapper class, in Hadoop 2.0
         * we can specify these values with conf.set (which is what I do) but I'm not sure if this is inefficient.
         */
        for(String stopWord: Files.readAllLines(Paths.get(args[0])))
            conf.set(stopWord, "1"); //set value 'stop word' to 1 if it is a stop word


        Job preprocessJob = Job.getInstance(conf, "SGML Raw Input Preprocessor");
        preprocessJob.setJarByClass(VectorSpaceRetrievalSystem.class);
        preprocessJob.setMapperClass(TokensMapper.class);
        preprocessJob.setReducerClass(PostingReducer.class);
        preprocessJob.setNumReduceTasks(1);
        preprocessJob.setMapOutputKeyClass(IntWritable.class);
        preprocessJob.setMapOutputValueClass(Text.class);
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

        Job freqJob = Job.getInstance(conf, "Calculate Frequencies");
        freqJob.setJarByClass(VectorSpaceRetrievalSystem.class);
        freqJob.setMapperClass(IdentityMapper.class);
        freqJob.setReducerClass(FrequencyReducer.class);
        freqJob.setNumReduceTasks(1);
        freqJob.setMapOutputKeyClass(Text.class);
        freqJob.setMapOutputValueClass(Text.class);
        freqJob.setOutputKeyClass(Text.class);
        freqJob.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(freqJob, preOutput);
        FileOutputFormat.setOutputPath(freqJob, freqOutput);

        int freqResult = freqJob.waitForCompletion(true) ? 0 : 1;

        if(freqResult == 1) {
            System.err.println("Something went wrong in the frequency calculating MapReduce job.");
            System.exit(preResult);
        }

        Job weightJob = Job.getInstance(conf, "Aggregate Weights");
        weightJob.setJarByClass(VectorSpaceRetrievalSystem.class);
        weightJob.setMapperClass(WeightMapper.class);
        weightJob.setReducerClass(WeightSumReducer.class);
        weightJob.setNumReduceTasks(1);
        weightJob.setMapOutputKeyClass(IntWritable.class);
        weightJob.setMapOutputValueClass(DoubleWritable.class);
        weightJob.setOutputKeyClass(IntWritable.class);
        weightJob.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(weightJob, freqOutput);
        FileOutputFormat.setOutputPath(weightJob, postOutput);

        int postResult = weightJob.waitForCompletion(true) ? 0 : 1;

        System.exit(postResult);
    }
}

