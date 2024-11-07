use application;
SET FOREIGN_KEY_CHECKS = 0;

alter table Customers modify id int UNSIGNED AUTO_INCREMENT;

SET FOREIGN_KEY_CHECKS = 1;

alter table Orders modify id int UNSIGNED AUTO_INCREMENT;

alter table Orders modify customer_id int UNSIGNED;

#ALTER TABLE `application`.`Orders` 
#DROP FOREIGN KEY `fk_Orders_1`;
#ALTER TABLE `application`.`Orders` 
#DROP INDEX `fk_Orders_1_idx` ;
#;

#ALTER TABLE `application`.`Orders` 
#ADD INDEX `fk_Orders_1_idx` (`customer_id` ASC) VISIBLE;
#;
#ALTER TABLE `application`.`Orders` 
#ADD CONSTRAINT `fk_Orders_1`
#  FOREIGN KEY (`customer_id`)
 # REFERENCES `application`.`Customers` (`id`)
 # ON DELETE NO ACTION
 # ON UPDATE NO ACTION;