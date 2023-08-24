import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import org.xml.sax.Attributes;
import org.xml.sax.helpers.DefaultHandler;
import java.io.File;
import static java.lang.System.exit;

/**
 * NetworkFactory class builds a Bayesian Network from an XML file. The XML file is expected to be in
 * topological order otherwise exception is raised.
 *
 * networkBuilder method returns a Bayesian Network object.
 * @author 220025456
 */
public class NetworkFactory
{
    public static BayesianNetwork networkBuilder(String uri)  {
        BayesianNetwork bn = new BayesianNetwork();
        try
        {
            SAXParserFactory factory = SAXParserFactory.newInstance();
            SAXParser saxParser = factory.newSAXParser();
            DefaultHandler handler = new DefaultHandler()
            {
                boolean tagFor = false;
                boolean tagGiven = false;
                boolean tagTable = false;
                Node node;

                //parser starts parsing a specific element inside the document
                public void startElement(String uri, String localName, String tagName, Attributes attributes)
                {
                    if(tagName.equalsIgnoreCase("for"))
                    {
                        tagFor = true;
                    }
                    if (tagName.equalsIgnoreCase("Given"))
                    {
                        tagGiven = true;
                    }
                    if (tagName.equalsIgnoreCase("Table"))
                    {
                        tagTable = true;
                    }
                }

                //reads the text value of the currently parsed element
                public void characters(char ch[], int start, int length)
                {
                    if (tagFor)
                    {
//                        System.out.println("Node : " + new String(ch, start, length));
                        node = bn.addNode(new String(ch, start, length));
                        tagFor = false;
                    }
                    if (tagGiven)
                    {
//                        System.out.println("Parent : " + new String(ch, start, length));
                        Node parentNode = bn.getNode(new String(ch, start, length));
                        node.addParent(parentNode);
                        parentNode.addChild(node);
                        tagGiven = false;
                    }
                    if (tagTable)
                    {
//                        System.out.println("Table : " + new String(ch, start, length));
                        node.addCPT(new String(ch, start, length));
                        tagTable = false;
                    }
                }
            };
            saxParser.parse(new File(uri), handler);

        }
        catch (NullPointerException npe)
        {
            System.out.println("non-DAG, NO topological order exists.");
            exit(1);
        } catch (Exception e)
        {
            e.printStackTrace();
            exit(1);
        }

        return bn;
    }
}  