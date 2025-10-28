import java.io.BufferedReader;
import java.io.FileReader;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.roi.RectangleROI
import qupath.lib.roi.ROIs
import qupath.lib.roi.EllipseROI;
import qupath.lib.geom.Point2
import qupath.lib.objects.*
import qupath.lib.objects.classes.*
import qupath.lib.objects.hierarchy.*
import qupath.lib.objects.hierarchy.events.*
import qupath.lib.regions.ImagePlane
import qupath.lib.objects.PathObjects
import qupath.lib.roi.RectangleROI
import qupath.lib.objects.PathAnnotationObject


def imageData = getCurrentImageData();

// Get location of csv
def file = "/Volumes/Finkbeiner-Steve/work/data/npsad_data/monika/ALS/tiles-npy/NEUKM699KKH_Lumbar_pTDP_43.csv"

// Create BufferedReader
def csvReader = new BufferedReader(new FileReader(file));

int sizePixels = 50
//int sizePixels = 120
row = csvReader.readLine() // first row (header)
color = getColorRGB(0, 100, 100)

// Loop through all the rows of the CSV file.
// Loop through all the rows of the CSV file.
while ((row = csvReader.readLine()) != null) {
    def rowContent = row.split(",")
    double r = rowContent[1] as double;
    //print r
    double c = rowContent[2] as double;
    //print c
    double x1 = rowContent[3] as double;
    //print x1
    double y1 = rowContent[4] as double;
    //print y1
    double w = rowContent[5] as double;
    //print w
    double h = rowContent[6] as double;
    double classid = rowContent[7];
    print classid
    cx = (r*512)+x1
    cy = (c*512)+y1
    // Create annotation
    def roi = new RectangleROI(cx, cy, w, h);
    // Create annotation
    //def roi = ROIs.createEllipseROI(cx-sizePixels/2,cy-sizePixels/2,sizePixels,sizePixels, ImagePlane.getDefaultPlane())
    //print roi
    //def annotation = new PathAnnotationObject(roi, PathClassFactory.getPathClass("Region"));
    //annotation.setColor(color)
    //def annotationExpansion = PathObjects.createAnnotationObject(roi,pathClass)
    //annotationExpansion.setName(itr +":"+ pathClass.toString())
    if (classid==49){
       classname="glia"
    }
    else {
        classname="TDP43"
}

    def annotation = new PathAnnotationObject(roi, PathClassFactory.getPathClass(classname));
    //imageData.getHierarchy().addPathObject(annotation, true);
    addObject(annotation)
}