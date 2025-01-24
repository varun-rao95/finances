# update account name

SELECT
  strftime ("%Y-%m", DATE) AS 'year-month',
  TransactionType,
  SUM(amount)
FROM
  transactions
WHERE
  AccountName = <accountname>  --  "Customized Cash Rewards Visa Signature - 4474", "Travel Rewards Visa Signature - 8166",...
GROUP BY
  1, 2
ORDER BY 
  1 ASC, 2 ASC;
