import os
import aiohttp
import asyncio
import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Literal, Dict, Optional, Any, Union
from langchain_core.tools import BaseTool, StructuredTool, tool, ToolException, InjectedToolArg
from langchain_core.messages import HumanMessage, AIMessage, MessageLikeRepresentation, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
import pandas as pd
import numpy as np
import json
import io
from pathlib import Path



# Set up logging for data tools
data_tools_logger = logging.getLogger('data_tools')
data_tools_logger.setLevel(logging.INFO)

# Create file handler for data tools logging
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = logging.FileHandler('logs/data_tools.log')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
data_tools_logger.addHandler(file_handler)

DOC_PATH = os.getenv("DOC_PATH", "./my-docs")

CSV_EXCEL_ANALYSIS_DESCRIPTION = (
    "A comprehensive tool for reading, analyzing, and manipulating CSV and Excel files using pandas and standard Python operations. Use this to understand the private data files not available on the web."
    "Supports data loading, filtering, aggregation, keyword search, statistical analysis, and data transformation operations. Only mention the name of the file, since all the files are inside a single directory."
)

@tool(description=CSV_EXCEL_ANALYSIS_DESCRIPTION)
async def csv_excel_analysis(
    file_name: str,
    operation: Literal[
        "read_data", "info", "head", "tail", "describe", 
        "filter", "sort", "groupby", "search"
    ],
    operation_params: Optional[Dict[str, Any]] = None,
    config: RunnableConfig = None
) -> str:
    """
    Perform various operations on CSV/Excel files using pandas.
    
    Args:
        file_name (str): Name of the CSV or Excel file in the data directory
        operation (str): The operation to perform on the data
        operation_params (dict): Parameters specific to the operation
        config (RunnableConfig): Configuration object (optional)
    
    Returns:
        str: Results of the operation in a formatted string
    """
    # Log the tool call
    log_data = {
        "tool": "csv_excel_analysis",
        "file_name": file_name,
        "operation": operation,
        "operation_params": operation_params or {},
        "timestamp": datetime.now().isoformat(),
        "config_keys": list(config.keys()) if config else []
    }
    data_tools_logger.info(f"TOOL_CALL: {json.dumps(log_data, indent=2)}")
    
    try:
        # Read the file based on its extension
        file_path = Path(DOC_PATH) / file_name
        if not file_path.exists():
            error_msg = f"Error: File '{file_path}' does not exist."
            data_tools_logger.error(f"FILE_NOT_FOUND: {file_path}")
            return error_msg
        
        # Determine file type and read accordingly
        if file_path.suffix.lower() in ['.csv']:
            # Read CSV with smart header detection
            df = pd.read_csv(file_path)
            
            # Check if columns are unnamed (pandas default for missing headers)
            unnamed_columns = [col for col in df.columns if 'Unnamed:' in str(col)]
            
            # If we have unnamed columns, try reading from second row (skip first row)
            if unnamed_columns:
                data_tools_logger.info(f"UNNAMED_COLUMNS_DETECTED: {file_path} - Found {len(unnamed_columns)} unnamed columns")
                
                # Read CSV again using second row as header (skip first row)
                df = pd.read_csv(file_path, header=1)
                data_tools_logger.info(f"USING_SECOND_ROW_AS_HEADER: {file_path} - Columns: {list(df.columns)}")
            
            data_tools_logger.info(f"FILE_READ: {file_path} (CSV) - Shape: {df.shape} - Columns: {list(df.columns)}")
            
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            # For Excel files, try to find the best sheet and header row
            try:
                # First try reading with default settings
                df = pd.read_excel(file_path)
                
                # Check if columns are unnamed (pandas default for missing headers)
                unnamed_columns = [col for col in df.columns if 'Unnamed:' in str(col)]
                
                # If we have unnamed columns, try reading from second row (skip first row)
                if unnamed_columns:
                    data_tools_logger.info(f"UNNAMED_COLUMNS_DETECTED_EXCEL: {file_path} - Found {len(unnamed_columns)} unnamed columns")
                    
                    # Read Excel again using second row as header (skip first row)
                    df = pd.read_excel(file_path, header=1)
                    data_tools_logger.info(f"USING_SECOND_ROW_AS_HEADER_EXCEL: {file_path} - Columns: {list(df.columns)}")
                
                data_tools_logger.info(f"FILE_READ: {file_path} (Excel) - Shape: {df.shape} - Columns: {list(df.columns)}")
                
            except Exception as e:
                data_tools_logger.error(f"EXCEL_READ_ERROR: {file_path} - Error: {str(e)}")
                # Try reading without header as fallback
                df = pd.read_excel(file_path, header=None)
                data_tools_logger.info(f"FILE_READ_FALLBACK: {file_path} (Excel) - Shape: {df.shape} - No header used")
        else:
            error_msg = f"Error: Unsupported file format '{file_path.suffix}'. Supported formats: .csv, .xlsx, .xls"
            data_tools_logger.error(f"UNSUPPORTED_FORMAT: {file_path.suffix}")
            return error_msg
        
        # Perform the requested operation
        result = await _execute_data_operation(df, operation, operation_params or {})
        
        # Log successful operation
        data_tools_logger.info(f"OPERATION_SUCCESS: {operation} on {file_path} - Result length: {len(str(result))}")
        return result
        
    except Exception as e:
        error_msg = f"Error performing operation '{operation}' on file '{file_path}': {str(e)}"
        data_tools_logger.error(f"OPERATION_ERROR: {operation} on {file_path} - Error: {str(e)}")
        return error_msg


async def _execute_data_operation(df: pd.DataFrame, operation: str, params: Dict[str, Any]) -> str:
    """Execute the specified data operation on the DataFrame."""
    
    try:
        if operation == "read_data":
            return _format_dataframe_info(df, "Data Overview")
            
        elif operation == "info":
            buffer = io.StringIO()
            df.info(buf=buffer, max_cols=None, memory_usage=True)
            return f"DataFrame Information:\n{buffer.getvalue()}"
            
        elif operation == "head":
            n = params.get("n", 5)
            return f"First {n} rows:\n{df.head(n).to_string()}"
            
        elif operation == "tail":
            n = params.get("n", 5)
            return f"Last {n} rows:\n{df.tail(n).to_string()}"
            
        elif operation == "describe":
            return f"Statistical Summary:\n{df.describe().to_string()}"
            
        elif operation == "filter":
            condition = params.get("condition")
            if not condition:
                return "Error: 'condition' parameter is required for filter operation"
            
            try:
                # Evaluate the condition safely
                filtered_df = df.query(condition)
                return f"Filtered data (condition: {condition}):\n{filtered_df.to_string()}\n\nShape: {filtered_df.shape}"
            except Exception as e:
                return f"Error evaluating filter condition '{condition}': {str(e)}"
                
        elif operation == "sort":
            by = params.get("by", df.columns[0])
            ascending = params.get("ascending", True)
            sorted_df = df.sort_values(by=by, ascending=ascending)
            return f"Sorted data (by: {by}, ascending: {ascending}):\n{sorted_df.head(10).to_string()}"
            
        elif operation == "groupby":
            group_cols = params.get("group_cols", [])
            agg_cols = params.get("agg_cols", [])
            agg_funcs = params.get("agg_funcs", ["mean"])
            
            if not group_cols:
                return "Error: 'group_cols' parameter is required for groupby operation"
            
            grouped = df.groupby(group_cols)
            if agg_cols and agg_funcs:
                result = grouped[agg_cols].agg(agg_funcs)
            else:
                result = grouped.size()
            
            return f"Grouped data:\n{result.to_string()}"
            
        elif operation == "search":
            search_term = params.get("search_term")
            column = params.get("column")
            case_sensitive = params.get("case_sensitive", False)
            partial_match = params.get("partial_match", True)
            
            # Log search operation details
            search_log_data = {
                "operation": "search",
                "search_term": search_term,
                "column": column,
                "case_sensitive": case_sensitive,
                "partial_match": partial_match,
                "dataframe_shape": df.shape,
                "available_columns": list(df.columns)
            }
            data_tools_logger.info(f"SEARCH_OPERATION: {json.dumps(search_log_data, indent=2)}")
            
            if not search_term:
                data_tools_logger.error("SEARCH_ERROR: Missing search_term parameter")
                return "Error: 'search_term' parameter is required for search operation"
            
            try:
                # If column is specified, search only in that column
                if column:
                    if column not in df.columns:
                        return f"Error: Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}"
                    
                    # Search in specific column
                    if case_sensitive:
                        if partial_match:
                            mask = df[column].astype(str).str.contains(search_term, na=False)
                        else:
                            mask = df[column].astype(str) == search_term
                    else:
                        if partial_match:
                            mask = df[column].astype(str).str.contains(search_term, case=False, na=False)
                        else:
                            mask = df[column].astype(str).str.lower() == search_term.lower()
                else:
                    # Search across all columns
                    mask = pd.Series([False] * len(df))
                    for col in df.columns:
                        if case_sensitive:
                            if partial_match:
                                col_mask = df[col].astype(str).str.contains(search_term, na=False)
                            else:
                                col_mask = df[col].astype(str) == search_term
                        else:
                            if partial_match:
                                col_mask = df[col].astype(str).str.contains(search_term, case=False, na=False)
                            else:
                                col_mask = df[col].astype(str).str.lower() == search_term.lower()
                        mask = mask | col_mask
                
                # Filter the DataFrame
                search_results = df[mask]
                
                # Log search results
                data_tools_logger.info(f"SEARCH_RESULTS: '{search_term}' - Found {len(search_results)} matches out of {len(df)} total rows")
                
                # Generate search summary
                total_matches = len(search_results)
                search_summary = f"Search Results for '{search_term}':\n"
                search_summary += f"Total matches found: {total_matches}\n"
                
                if column:
                    search_summary += f"Searching in column: '{column}'\n"
                else:
                    search_summary += "Searching across all columns\n"
                
                search_summary += f"Case sensitive: {case_sensitive}\n"
                search_summary += f"Partial match: {partial_match}\n"
                search_summary += f"Total rows in dataset: {len(df)}\n"
                search_summary += f"Match percentage: {(total_matches/len(df)*100):.1f}%\n\n"
                
                if total_matches == 0:
                    search_summary += "No matches found."
                else:
                    # Show all results (or limit if too many)
                    max_display = params.get("max_display", 50)
                    if total_matches > max_display:
                        search_summary += f"Showing first {max_display} results (out of {total_matches}):\n\n"
                        display_results = search_results.head(max_display)
                    else:
                        search_summary += "All matching results:\n\n"
                        display_results = search_results
                    
                    search_summary += display_results.to_string(index=False)
                    
                    if total_matches > max_display:
                        search_summary += f"\n\n... and {total_matches - max_display} more results"
                
                return search_summary
                
            except Exception as e:
                return f"Error performing search: {str(e)}"
            
        else:
            return f"Error: Unknown operation '{operation}'"
            
    except Exception as e:
        return f"Error executing operation '{operation}': {str(e)}"


def _format_dataframe_info(df: pd.DataFrame, title: str) -> str:
    """Format DataFrame information in a readable way."""
    info = f"{title}\n"
    info += f"Shape: {df.shape} (rows, columns)\n"
    info += f"Columns: {list(df.columns)}\n"
    info += f"Data types:\n{df.dtypes.to_string()}\n"
    info += f"First 5 rows:\n{df.head().to_string()}\n"
    
    # Add basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info += f"\nNumeric columns summary:\n{df[numeric_cols].describe().to_string()}"
    
    return info


##########################
# List Data Files Tool
##########################
LIST_DATA_FILES_DESCRIPTION = (
    "A tool for exploring and listing all files in a data folder. Use this to discover available datasets before using the CSV/Excel analysis tool. Supports filtering by file type and provides file information."
)

@tool(description=LIST_DATA_FILES_DESCRIPTION)
async def list_data_files(
    folder_path: str = DOC_PATH,
    file_types: Optional[List[str]] = None,
    include_subfolders: bool = False,
    show_details: bool = True,
    config: RunnableConfig = None
) -> str:
    """
    List all files in a specified data folder with optional filtering and detailed information.
    
    Args:
        folder_path (str): Path to the folder to list files from (default: "data")
        file_types (List[str]): Optional list of file extensions to filter by (e.g., [".csv", ".xlsx"])
        include_subfolders (bool): Whether to include files from subfolders (default: False)
        show_details (bool): Whether to show file details like size and modification date (default: True)
        config (RunnableConfig): Configuration object (optional)
    
    Returns:
        str: Formatted list of files with details
    """
    # Log the tool call
    log_data = {
        "tool": "list_data_files",
        "folder_path": folder_path,
        "file_types": file_types,
        "include_subfolders": include_subfolders,
        "show_details": show_details,
        "timestamp": datetime.now().isoformat(),
        "config_keys": list(config.keys()) if config else []
    }
    data_tools_logger.info(f"TOOL_CALL: {json.dumps(log_data, indent=2)}")
    
    try:
        folder_path = Path(folder_path)
        
        # Check if folder exists
        if not folder_path.exists():
            error_msg = f"Error: Folder '{folder_path}' does not exist."
            data_tools_logger.error(f"FOLDER_NOT_FOUND: {folder_path}")
            return error_msg
        
        if not folder_path.is_dir():
            error_msg = f"Error: '{folder_path}' is not a directory."
            data_tools_logger.error(f"NOT_A_DIRECTORY: {folder_path}")
            return error_msg
        
        # Define supported file types for data analysis
        supported_types = {
            '.csv': 'CSV Data File',
            '.xlsx': 'Excel Spreadsheet',
            '.xls': 'Excel Spreadsheet (Legacy)',
            '.json': 'JSON Data File',
            '.parquet': 'Parquet Data File',
            '.feather': 'Feather Data File',
            '.h5': 'HDF5 Data File',
            '.hdf5': 'HDF5 Data File',
            '.pkl': 'Pickle Data File',
            '.pickle': 'Pickle Data File',
            '.txt': 'Text Data File',
            '.tsv': 'Tab-Separated Values',
            '.dat': 'Data File',
            '.db': 'Database File',
            '.sqlite': 'SQLite Database',
            '.sql': 'SQL Script'
        }
        
        # Collect files
        files = []
        if include_subfolders:
            for file_path in folder_path.rglob("*"):
                if file_path.is_file():
                    files.append(file_path)
        else:
            for file_path in folder_path.iterdir():
                if file_path.is_file():
                    files.append(file_path)
        
        # Filter by file type if specified
        if file_types:
            filtered_files = []
            for file_path in files:
                if file_path.suffix.lower() in [ext.lower() for ext in file_types]:
                    filtered_files.append(file_path)
            files = filtered_files
        
        if not files:
            if file_types:
                return f"No files found in '{folder_path}' with specified types: {file_types}"
            else:
                return f"No files found in '{folder_path}'"
        
        # Sort files by name
        files.sort(key=lambda x: x.name.lower())
        
        # Generate report
        report = f"Files found in '{folder_path}':\n"
        report += f"Total files: {len(files)}\n"
        
        if file_types:
            report += f"Filtered by types: {file_types}\n"
        
        if include_subfolders:
            report += "Including subfolders: Yes\n"
        
        report += "\n" + "="*80 + "\n\n"
        
        # Group files by type
        files_by_type = {}
        for file_path in files:
            file_type = file_path.suffix.lower()
            if file_type not in files_by_type:
                files_by_type[file_type] = []
            files_by_type[file_type].append(file_path)
        
        # Sort file types
        sorted_types = sorted(files_by_type.keys())
        
        for file_type in sorted_types:
            type_files = files_by_type[file_type]
            type_name = supported_types.get(file_type, f"{file_type.upper()} File")
            
            report += f"## {type_name} ({len(type_files)} files)\n\n"
            
            for file_path in type_files:
                report += f"**{file_path.name}**\n"
                
                if show_details:
                    try:
                        # Get file stats
                        stat = file_path.stat()
                        size_bytes = stat.st_size
                        modified_time = datetime.fromtimestamp(stat.st_mtime)
                        
                        # Format file size
                        if size_bytes < 1024:
                            size_str = f"{size_bytes} B"
                        elif size_bytes < 1024**2:
                            size_str = f"{size_bytes/1024:.1f} KB"
                        elif size_bytes < 1024**3:
                            size_str = f"{size_bytes/1024**2:.1f} MB"
                        else:
                            size_str = f"{size_bytes/1024**3:.1f} GB"
                        
                        # Get relative path
                        rel_path = file_path.relative_to(folder_path)
                        
                        report += f"  - Path: {rel_path}\n"
                        report += f"  - Size: {size_str}\n"
                        report += f"  - Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        
                        # Add preview for supported file types
                        if file_type in ['.csv', '.xlsx', '.xls']:
                            try:
                                if file_type == '.csv':
                                    df = pd.read_csv(file_path, nrows=3)
                                else:
                                    df = pd.read_excel(file_path, nrows=3)
                                
                                report += f"  - Preview: {df.shape[0]} rows, {df.shape[1]} columns\n"
                                report += f"  - Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}\n"
                            except Exception as e:
                                report += f"  - Preview: Unable to read file ({str(e)[:50]}...)\n"
                        
                    except Exception as e:
                        report += f"  - Error getting file details: {str(e)}\n"
                
                report += "\n"
        
        # Add summary statistics
        report += "## Summary\n\n"
        report += f"- Total files: {len(files)}\n"
        report += f"- File types found: {len(files_by_type)}\n"
        
        # File type breakdown
        type_breakdown = []
        for file_type, type_files in files_by_type.items():
            type_name = supported_types.get(file_type, file_type.upper())
            type_breakdown.append(f"{type_name}: {len(type_files)}")
        
        report += f"- File type breakdown: {', '.join(type_breakdown)}\n"
        
        # Total size
        total_size = 0
        for file_path in files:
            try:
                total_size += file_path.stat().st_size
            except:
                pass
        
        if total_size > 0:
            if total_size < 1024**2:
                total_size_str = f"{total_size/1024:.1f} KB"
            elif total_size < 1024**3:
                total_size_str = f"{total_size/1024**2:.1f} MB"
            else:
                total_size_str = f"{total_size/1024**3:.1f} GB"
            report += f"- Total size: {total_size_str}\n"
        
        # Log successful operation
        data_tools_logger.info(f"LIST_FILES_SUCCESS: {folder_path} - Found {len(files)} files")
        return report
        
    except Exception as e:
        error_msg = f"Error listing files in '{folder_path}': {str(e)}"
        data_tools_logger.error(f"LIST_FILES_ERROR: {folder_path} - Error: {str(e)}")
        return error_msg
    

from langgraph.prebuilt import create_react_agent
from langchain_openai import OpenAI
tools = [list_data_files, csv_excel_analysis]

from langchain_core.prompts import PromptTemplate
template = """
You are a helpful assistant that can help with data analysis and manipulation. You are part of a team of researchers that are working on a project. You will be given a query, and your job is use the tools you have to understand the data files you have access to deeply, and retrieve any information that is relevant to the query.
You should not end the call until you have found something relevant to the query explicitly. You should use the search/filter tools to find the information you need, after understanding the structure of the data files.
"""

model = OpenAI()
agent = create_react_agent(tools=tools, model="gpt-4o-mini", prompt=template)






