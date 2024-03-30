package com.a303.consultms.domain.consult.repository;

import com.a303.consultms.domain.consult.Consult;
import com.a303.consultms.domain.consult.ConsultCategory;
import com.a303.consultms.domain.consult.dto.response.ConsultDetailRes;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ConsultCategoryRepository extends JpaRepository<ConsultCategory, Integer> {
}
