package news_java_ner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.sequences.DocumentReaderAndWriter;
import edu.stanford.nlp.util.Triple;

public class News_ner {
	public static BufferedWriter writetoFile(String filename) throws IOException{
		File file = new File(filename);
		
		if (!file.exists()) {
				file.createNewFile();
		}
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		return bw;
		
	}
	
	public static void nerFiles(String directory_,String output_,AbstractSequenceClassifier<CoreLabel> classifier) throws IOException{
		//Read from the folder
		File folder = new File(directory_);
		File[] listFiles = folder.listFiles();
		String line ="";
		String input = "";
		BufferedWriter bw = null;
		for (File file:listFiles){
			
			if (file.isFile()){
					input ="";
					FileReader fileReader = new FileReader(file);
					BufferedReader bufferedReader = new BufferedReader(fileReader);
					bw= writetoFile(output_+"/"+file.getName());
					while((line = bufferedReader.readLine())!=null){
						input+=line;
						
					}
					bufferedReader.close(); 
					  
			// write each sentence w/ NER tags to another file
					bw.write(classifier.classifyToString(input,"inlineXML",false)); 
					bw.close();
			}
		}
		
		//classifier.classifyToString(str, "slashTags", false)
	   
	}
	public static BufferedReader FileReader(String filename) throws FileNotFoundException{
		File file = new File(filename);
		FileReader fileReader = new FileReader(file);
		BufferedReader bufferedReader = new BufferedReader(fileReader);
		return bufferedReader;
	}
	
	public static void main(String[] args) throws Exception{
		//use 7class CRF classifier
		
		String serializedClassifier = "classifiers/english.muc.7class.distsim.crf.ser.gz";
		if (args.length > 0) {
		      serializedClassifier = args[0];
		}
		//initiate the classifer
		AbstractSequenceClassifier<CoreLabel> classifier = CRFClassifier.getClassifier(serializedClassifier);
		//Initiate a bufferedReader
		
		nerFiles("WP/","WP_output",classifier);
		
		
		//BufferedReader bufferedReader = FileReader("University Wire_titles_.txt");
		//BufferedWriter bw = writetoFile("title1.txt");;
		
		
		BufferedReader bufferedReader = FileReader("washingtonpost_titles_.txt");
		BufferedWriter bw = writetoFile("title.txt");;		
		
		//read the file
		String line=null;
		int index=0;
		while((line = bufferedReader.readLine())!=null){
			//String result=classifier.classifyToString(line,"inlineXML",false);
			
			List<Triple<String,Integer,Integer>> triples = classifier.classifyToCharacterOffsets(line);
	        for (Triple<String,Integer,Integer> trip : triples) {
	          String result = trip.first()+":"+line.substring(trip.second, trip.third);
	          
	          if (result!=null){
	          String[] listStr = result.split("\n");
	          bw.write('\n');
	          bw.write(String.valueOf(index));
	          bw.write('\n');
	          for(String a:listStr){
	        	  System.out.println(index);
	        	  System.out.println(a);
	        	  bw.write(a+'\n');
	          }
	          }
	        }  
			index+=1;
			
		}
		bufferedReader.close(); 
		bw.close();
}		
}
