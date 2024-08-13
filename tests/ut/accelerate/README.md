## How to run accelerate UTs? 

Once inside the docker container, you can use the following commands to run accelerate tests

```bash
cd accelerate  
export REPORT=/mnt/accelerate/ut_results.xlsx
export LOG=/mnt/accelerate/ut_results.log
RUN_SLOW=1 python -m pytest tests -sv --excelreport $REPORT 2>&1 | tee $LOG
```
