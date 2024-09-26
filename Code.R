#STEP 1 Import necessary libraries
library(shiny)
library(shinythemes)
library(data.table)
library(caret)
library(randomForest)
library(flexdashboard)
library(shinyLP)
library(psych)
library(GGally)
library(shinyWidgets)
library(rpart)
library(DT)
library(rpart.plot)
library(ggplot2)

#--------------------------------------------XXXXXXX--------------------------------

#STEP 2 Read the dataset from a CSV file
data <- read.csv("/Users/Group 15/Downloads/employee_attrition.csv", header = TRUE)

# Preparing the dataset for machine learning
# Set a seed for reproducible random splits
set.seed(123)  

#--------------------------------------------XXXXXXX--------------------------------

# STEP 3 Split the data into training and testing sets (80% training, 20% testing)
split_indices <- createDataPartition(data$Attrition, p = 0.8, list = FALSE)
train_set <- data[split_indices, ]
test_set <- data[-split_indices, ]

# Separate the features and target variable for both training and testing sets
X_train <- train_set[, -which(names(train_set) == "Attrition")]
y_train <- train_set$Attrition
X_test <- test_set[, -which(names(test_set) == "Attrition")]
y_test <- test_set$Attrition

# Convert target variable to factor for classification
y_train <- factor(y_train, levels = c(0, 1))
y_test <- factor(y_test, levels = c(0, 1))

#--------------------------------------------XXXXXXX--------------------------------

# STEP 4 Define the UI for the app with various panels and inputs
ui <- fluidPage(theme= shinytheme("flatly"), 
                navbarPage(title = "Group 15",  id="inTabset",
                           # Home panel UI setup
                           tabPanel("Home", style = "text-align: center;",
                                    h1("HELLO, LET'S PREDICT EMPLOYEE ATTRITION"),
                                    # Description and navigation button
                                    div(style = "margin-bottom: 50px;"),
                                    strong(h4("This tool will help you to predict employee attrition using Decision Tree and Random Forest ML models. A brief description of the various tabs is given below.")),
                                    br(),
                                    DTOutput("tab_description_table"),
                                    br(),
                                    actionButton('next_tab',strong(h5('Click here to start',icon("paper-plane"))),
                                                 style="color: #fff; background-color: #051E32; border-color: 051E32")
                           ),
                           
                           # Summary statistics panel UI setup
                           tabPanel("Statistics",
                                    sidebarLayout(
                                      sidebarPanel(
                                        # File upload and variable selection input
                                        fileInput("file1", "Upload a CSV File", accept=c('text/csv', 'text/comma-separated-values', 'text/plain', '.csv')),
                                        selectInput(inputId = "cols1",label = "Choose Variable:", choices = "", selected = " ", multiple = TRUE),
                                        hr()
                                        
                                      ),
                                      mainPanel(
                                        # Output area for summary statistics
                                        verbatimTextOutput("summar_output")
                                      )
                                    )
                           ),
                           
                           # Visualization panel UI setup
                           tabPanel('Visualisation', style = "text-align: center;",
                                    headerPanel("Interactive Data Visualisation"),
                                    sidebarLayout(
                                      sidebarPanel(
                                        # Inputs for histogram and bar plot
                                        HTML("<h4>Histogram & Bar Plot</h4>"),
                                        selectInput("variable", "Select a variable:",
                                                    choices = names(data)),
                                        sliderInput("bins",
                                                    "Select number of bins for histogram:",
                                                    min = 1,
                                                    max = 50,
                                                    value = 30)
                                      ),
                                      mainPanel(
                                        # Plot output area
                                        plotOutput("plot")
                                      )   )),
                           
                           # Combined UI panel for Model (Random Forest and Decision Tree)
                           tabPanel('Model',
                                    headerPanel("Model Summary"),
                                    sidebarLayout(
                                      sidebarPanel(
                                        # Input controls for model parameters
                                        selectInput("selected_model", "Model", 
                                                    choices = c("Random Forest", "Decision Tree"),
                                                    selected = "Random Forest"),
                                        div(style = "margin-bottom: 25px;"),
                                        actionButton("run_model", "Run Model", 
                                                     style = 'color:black; padding:4px; font-size:110%; font-weight: bold;'),
                                        div(style = "margin-bottom: 30px;"),
                                        sliderInput("num_tree_model", "Number of trees (RF)", 
                                                    min = 1, max = 1000, value = 500),
                                        div(style = "margin-bottom: 30px;"),
                                        sliderInput("mtry_model", "Number of features at each split (RF)", 
                                                    min = 1, max = 35, value = 2),
                                        div(style = "margin-bottom: 30px;"),
                                        # Decision Tree specific controls
                                        sliderInput("cp_model", "Complexity Parameter (DT)", 
                                                    min = 0.001, max = 0.1, value = 0.01, step = 0.001),
                                        div(style = "margin-bottom: 30px;"),
                                        sliderInput("minsplit_model", "Minimum Samples for Split (DT)", 
                                                    min = 1, max = 20, value = 2)
                                      ),
                                      mainPanel(width = 8,
                                                fluidRow(
                                                  column(6, plotOutput("con_Matrix_model")),
                                                  column(6, tableOutput("metrics_Table_model")),
                                                  # Decision tree plot output conditional on model selection
                                                  conditionalPanel(
                                                    condition = "input.selected_model == 'Decision Tree'",
                                                    plotOutput("decisionTreePlot")
                                                  ),
                                                  # Random forest plot output conditional on model selection
                                                  conditionalPanel(
                                                    condition = "input.selected_model == 'Random Forest'",
                                                    plotOutput("randomForestPlot")))
                                      ))),
                           
                           # Prediction panel UI setup
                           tabPanel("Prediction",
                                    headerPanel('Attrition Predictor'),
                                    selectInput("selected_model_predict", "Select Model", choices = c("Random Forest", "Decision Tree")),
                                    div(style = "padding: 5px;", actionButton("predict_button", "Predict")),
                                    tableOutput("prediction_table"),
                                    div(style = "padding: 5px;", downloadButton("downloadButton", "Export Table"))
                           ),
                           # Contact information panel UI setup
                           tabPanel("Team",
                                    sidebarLayout(
                                      sidebarPanel(
                                        # Contact information
                                        h3("In case of any queries about this application, please contact the team")
                                      ),
                                      mainPanel(htmlOutput("text_output"))
                                    )
                           )
                ))

#--------------------------------------------XXXXXXX--------------------------------


#STEP 5      Server                    

# Define server logic for Shiny app
server<- function(input, output, session){
  
  # Home panel server logic
  # Listen for the 'next_tab' button click event to navigate to the "Statistics" tab
  observeEvent(input$next_tab, {
    updateTabsetPanel(session, inputId = "inTabset", selected = "Statistics")
  })
  
  # Data for the description table on the Home tab
  # This table describes the purpose of each tab in the app
  tab_description_data <- data.frame(
    Tab = c("Statistics", "Visualisation", "Model", "Prediction", "Contact"),
    Description = c(
      "Upload a CSV file and view summary statistics of the uploaded data.",
      "Create interactive data visualizations, including histograms and bar plots.",
      "Customise hyperparameters for clasification models.",
      "Upload unseen data and make predictions using the trained models.",
      "Get information to contact for further assistance."
    )
  )
  
  # Render the description table on the Home tab
  # This output renders a DataTable widget with the description of each app tab
  output$tab_description_table <- renderDT({
    datatable(
      tab_description_data, # Data source for the table
      options = list(
        pageLength = 5, # Number of rows to display (set to match number of tabs for simplicity)
        searching = FALSE, # Disable search box
        lengthChange = FALSE, # Disable ability to change number of rows displayed
        info = FALSE, # Disable table information (e.g., "Showing 1 to 5 of 5 entries")
        paging = FALSE, # Disable pagination
        columnDefs = list(list(className = 'dt-left', targets = c(0, 1))) # Align text in both columns to the left
      ),
      rownames = FALSE) # Do not display row names
  })
  

  # Summary Statistics Panel Server Logic
  
  # Reactive expression for reading uploaded data
  data_input <- reactive({
    infile <- input$file1 # Input from file upload UI component
    req(infile) # Require that the file input is not NULL
    read.csv(infile$datapath, header = TRUE, stringsAsFactors = FALSE) # Read the uploaded CSV file
  })
  
  # Observe event of file upload to update variable selection inputs
  observeEvent(input$file1, {
    # Update the choices for selectInput based on the columns of the uploaded data
    updateSelectInput(session, inputId = "cols", choices = names(data_input()))
    updateSelectInput(session, inputId = "cols1", choices = names(data_input()))
  })
  
  # Render the summary statistics of the selected variable(s)
  output$summar_output <- renderPrint({
    req(input$file1, input$cols1) # Require   at least one variable are selected
    
    var1 <- data_input()[, input$cols1] # Subset the data based on selected variable(s)
    
    if (!is.null(var1) && length(var1) > 0) {
      summary_stats <- summary(var1) # Calculate summary statistics
      return(summary_stats) # Return the summary statistics
    } else {
      return(NULL) # Return NULL if the variable is NULL or has length 0
    }
  })
  
  
  # Visualization Panel Server Logic
  
  # Reactive expression for handling uploaded data
  uploaded_data <- reactive({
    if (is.null(input$file1)) {
      return(NULL) # Return NULL if no file is uploaded
    }
    
    read.csv(input$file1$datapath, header = TRUE, stringsAsFactors = FALSE) # Read the uploaded CSV file
  })
  
  # Define a reactive expression for the selected variable
  selected_variable <- reactive({
    req(uploaded_data()) # Require that the uploaded data is not NULL
    variable <- uploaded_data()[[input$variable]] # Subset the uploaded data for the selected variable
    
    # Treat the variable as categorical if it has less than 6 unique values
    if (length(unique(variable)) < 6) {
      factor(variable)
    } else {
      variable
    }
  })
  
  # Reactive expression to check if the selected variable is categorical
  is_categorical <- reactive({
    is.factor(selected_variable()) || is.character(selected_variable())
  })
  
  # Render histogram or bar plot based on the type of the selected variable
  output$plot <- renderPlot({
    req(uploaded_data()) # Require that the uploaded data is not NULL
    
    if (is_categorical()) {
      # Render a bar plot for categorical variables using ggplot2
      ggplot(uploaded_data(), aes(x = factor(selected_variable()))) +
        geom_bar(fill = "skyblue", color = "black") +
        labs(title = "Bar Plot", x = input$variable) +
        theme(plot.title = element_text(hjust = 0.5))
    } else {
      # Render a histogram for continuous variables using ggplot2
      ggplot(uploaded_data(), aes(x = selected_variable())) +
        geom_histogram(fill = "lightcoral", color = "black", bins = input$bins) +
        labs(title = "Histogram", x = input$variable) +
        theme(plot.title = element_text(hjust = 0.5))
    }
  })
  
  # Train Random Forest and Decision Tree models panel server logic
  
  # Reactive expression to train models upon button click
  trained_models <- reactive({
    req(input$run_model)  # Ensure 'Run Model' button was clicked
    
    # Check which model is selected and train accordingly
    if (input$selected_model == "Random Forest") {
      # Train a Random Forest model with specified parameters
      set.seed(123)  # Ensure reproducibility
      model <- randomForest(factor(y_train) ~ ., data = X_train,
                            importance = TRUE, ntree = input$num_tree_model,
                            mtry = input$mtry_model, do.trace = 100, proximity = TRUE)
      y_pred <- predict(model, newdata = X_test)  # Make predictions
      y_pred <- factor(y_pred, levels = levels(factor(y_test)))  # Ensure factor levels match
      
    } else {
      # Train a Decision Tree model with specified control parameters
      set.seed(123)  # Ensure reproducibility
      
      y_test <- factor(y_test, levels = levels(factor(y_train)))  # Align factor levels
      model <- rpart(factor(y_train) ~ ., data = X_train, method="class",
                     control=rpart.control(cp = input$cp_model, minsplit = input$minsplit_model))
      y_pred_probs <- predict(model, newdata = X_test, type = "prob")  # Predict probabilities
      
      y_pred <- as.factor(ifelse(y_pred_probs[, "1"] > 0.5, "1", "0"))  # Determine class based on probability
      y_pred <- factor(y_pred, levels = levels(factor(y_test)))  # Ensure factor levels match
    }
    
    # Return the trained model and predictions
    return(list(model = model, y_pred = y_pred))
  })
  
  # Render the confusion matrix plot
  output$con_Matrix_model <- renderPlot({
    req(trained_models())  # Ensure model has been trained
    
    y_pred <- trained_models()$y_pred  # Extract predictions
    confmatrix <- confusionMatrix(factor(y_test), y_pred)  # Compute confusion matrix
    cmatrix_data <- as.data.frame(table(Predicted = y_pred, Actual = factor(y_test)))  # Convert to data frame
    
    # Plot confusion matrix using ggplot2
    ggplot(cmatrix_data) +
      geom_tile(aes(x = Actual, y = Predicted, fill = Freq), color = 'white') +
      geom_text(aes(x = Actual, y = Predicted, label = sprintf("%d", Freq)), vjust = 1) +
      scale_fill_gradient(low = "white", high = "cyan") +
      theme_minimal() +
      labs(fill = "Frequency", x = "Actual Label", y = "Predicted Label") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  # Render the table of model performance metrics
  output$metrics_Table_model <- renderTable({
    req(trained_models())  # Ensure model has been trained
    
    y_pred <- trained_models()$y_pred  # Extract predictions
    confmatrix <- confusionMatrix(factor(y_test), y_pred)  # Compute confusion matrix
    
    # Calculate performance metrics
    precision <- confmatrix$byClass['Pos Pred Value']
    recall <- confmatrix$byClass['Sensitivity']
    f1 <- confmatrix$byClass['F1']
    accuracy <- confmatrix$overall['Accuracy']
    
    # Create a table with the metrics
    metrics_data <- data.frame(Precision = precision, Recall = recall, F1_Score = f1, Accuracy = accuracy)
    metrics_data  # Return the data frame for rendering
  })
  
  # Decision Tree Plot
  output$decisionTreePlot <- renderPlot({
    req(input$run_model, input$selected_model == "Decision Tree")
    model <- trained_models()$model
    rpart.plot(model)
  }, width = 800, height = 600) 
  
  # Random Forest Feature Importance Plot
  output$randomForestPlot <- renderPlot({
    req(input$run_model, input$selected_model == "Random Forest")
    model <- trained_models()$model
    
    # Extract variable importance for MeanDecreaseGini
    varImpGini <- varImp(model, type = 2)  # type = 2 for MeanDecreaseGini
    
    # Convert to data frame for ggplot
    varImpGiniDf <- as.data.frame(varImpGini)
    varImpGiniDf$Variable <- rownames(varImpGiniDf)
    
    # Plot using ggplot2
    ggplot(varImpGiniDf, aes(x = reorder(Variable, Overall), y = Overall)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      xlab("Variables") +
      ylab("Mean Decrease Gini") +
      coord_flip() +  # Flip coordinates for horizontal bars
      theme_minimal() +
      ggtitle("Variable Importance (MeanDecreaseGini)")
  }, width = 800, height = 600)
  
  ########################################################
  # Prediction panel server logic
  
  # Listen for the 'Predict' button click event
  observeEvent(input$predict_button, {
    # Ensure that a model is selected, models are trained, and data is uploaded
    req(input$selected_model_predict, trained_models(), uploaded_data())
    
    # Retrieve the selected model type and the actual model object
    selected_model <- input$selected_model_predict
    model <- trained_models()$model
    
    # Prediction logic varies based on the selected model type
    if (selected_model == "Random Forest") {
      # For Random Forest, predict probabilities and then classify as '1' or '0'
      y_pred_probs <- predict(model, newdata = uploaded_data(), type = "prob")
      y_pred <- as.factor(ifelse(y_pred_probs > 0.5, "1", "0"))
      
    } else {
      # For Decision Tree, predict probabilities and classify based on a threshold
      y_pred_probs <- predict(model, newdata = uploaded_data(), type = "prob")
      y_pred <- as.factor(ifelse(y_pred_probs[, "1"] > 0.5, "1", "0"))
    }
    
    # Combine the predictions with the uploaded data
    updated_data <- cbind(uploaded_data(), Predicted = y_pred)
    
    # Store the updated data with predictions for display and export
    exported_data <- reactiveVal(updated_data)
    
    # Render a table to display the uploaded data along with the predictions
    output$prediction_table <- renderTable({
      exported_data()
    })
    
    # Setup a download handler to allow users to export the prediction results
    output$downloadButton <- downloadHandler(
      filename = function() {
        # Generate a filename including the current date
        paste("exported_data_", Sys.Date(), ".csv", sep = "")
      },
      content = function(file) {
        # Write the prediction results to a CSV file
        write.csv(exported_data(), file, row.names = FALSE)
      }
    )
  })
  
  
  
  #########################################################
  # Render contact information dynamically in the UI
  output$text_output <- renderUI({
    # Concatenate the team member's name and email address into HTML strings
    str1 <- paste(h3("Nikhil Rawat"))
    str2 <- paste(h4("E-mail: Nikhil.Rawat@bayes.city.ac.uk"))
    # Visual separator
    str3 = paste('*****************************************************************************************************')
    
    str4 <- paste(h3("Pooja Amarnani"))
    str5 <- paste(h4("E-mail: Pooja.Amarnani@bayes.city.ac.uk"))
    # Visual separator
    str6 = paste('*****************************************************************************************************')
    
    str7 <- paste(h3("Sista Sai"))
    str8 <- paste(h4("E-mail: Sai.Sista@bayes.city.ac.uk"))
    # Visual separator
    str9 = paste('*****************************************************************************************************')
    
    str10 <- paste(h3("Shubham Verma"))
    str11 <- paste(h4("E-mail: Shubham.Verma.2@bayes.city.ac.uk"))
    
    # Combine all strings into one HTML block, separated by line breaks
    HTML(paste(str1, str2, str3, str4, str5, str6, str7, str8, str9, str10, str11, sep = '<br/>'))
 })
}
  

#--------------------------------------------XXXXXXX--------------------------------


# STEP 6 Create the shiny app             
shinyApp(ui = ui, server = server)


  

